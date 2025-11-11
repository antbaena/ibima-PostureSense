#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os

# Mensajes
from multicam_cube_calib_interfaces.msg import CameraPairPose
from geometry_msgs.msg import PoseWithCovariance, TransformStamped, Transform
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage


# =======================
#  Utilidades SO(3)/SE(3)
# =======================

def _hat3(w):
    x, y, z = w
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=float)

def _vee3(W):
    return np.array([W[2,1]-W[1,2], W[0,2]-W[2,0], W[1,0]-W[0,1]])*0.5

def so3_exp(phi):
    theta = np.linalg.norm(phi)
    W = _hat3(phi)
    if theta < 1e-12:
        A = 1 - theta**2/6 + theta**4/120
        B = 0.5 - theta**2/24 + theta**4/720
    else:
        A = np.sin(theta)/theta
        B = (1 - np.cos(theta))/theta**2
    return np.eye(3) + A*W + B*(W@W)

def so3_log(R):
    c = (np.trace(R)-1)/2
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        d = np.diag(R)
        k = int(np.argmax(d))
        v = np.zeros(3)
        v[k] = np.sqrt(max(0.0, (d[k]+1)/2))
        j, l = (k+1) % 3, (k+2) % 3
        v[j] = (R[j,k]+R[k,j])/(4*v[k] + 1e-12)
        v[l] = (R[l,k]+R[k,l])/(4*v[k] + 1e-12)
        v = v/np.linalg.norm(v)
        return theta*v
    W = (R - R.T)/(2*np.sin(theta))
    return theta*_vee3(W)

def _left_jac_SO3_inv(phi):
    theta = np.linalg.norm(phi)
    I = np.eye(3)
    if theta < 1e-8:
        W = _hat3(phi)
        return I + 0.5*W + (1/12)*(W@W)
    half = 0.5*theta
    cot_half = np.cos(half)/np.sin(half)
    W = _hat3(phi)
    return I - 0.5*W + (1 - theta*cot_half)/(theta**2) * (W@W)

def se3_exp(xi):
    rho = np.asarray(xi[:3]); phi = np.asarray(xi[3:])
    R = so3_exp(phi)
    theta = np.linalg.norm(phi)
    I = np.eye(3); W = _hat3(phi)
    if theta < 1e-12:
        V = I + 0.5*W + (1/6)*(W@W)
    else:
        V = I + (1-np.cos(theta))/theta**2 * W + (theta-np.sin(theta))/theta**3 * (W@W)
    t = V @ rho
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    return T

def se3_log(T):
    R = T[:3,:3]; t = T[:3,3]
    phi = so3_log(R)
    V_inv = _left_jac_SO3_inv(phi)
    rho = V_inv @ t
    return np.r_[rho, phi]

def adjoint_SE3(T):
    R = T[:3,:3]; t = T[:3,3]
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = _hat3(t) @ R
    return Ad

# Quaternion/Rot-mat helpers (sin dependencias externas)
def quat_to_rotmat(qx, qy, qz, qw):
    # asegura normalización
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=float)
    return R

def rotmat_to_quat(R):
    # devuelve (x,y,z,w)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
            qw = (R[2,1] - R[1,2]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
            qw = (R[0,2] - R[2,0]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
            qw = (R[1,0] - R[0,1]) / S
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)
    return q

# Pérdida robusta
def _robust_weight(norm, c=1.345, kind='huber'):
    z = norm/c if c > 0 else norm
    if kind == 'huber':
        return 1.0 if norm <= c else (c/(norm+1e-12))
    if kind == 'cauchy':
        return 1.0/(1.0+z*z)
    if kind == 'tukey':
        return (1 - z*z)**2 if abs(z) < 1 else 0.0
    return 1.0  # none


def se3_average(transforms: List[np.ndarray],
                covariances: List[np.ndarray],
                loss: str = 'huber',
                c: float = 1.5,
                max_iters: int = 60,
                tol: float = 1e-9,
                meters_per_radian: Optional[float] = None
               ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estima la media robusta en SE(3) (MLE con M-estimador).
    """
    Ts = [np.asarray(T, float) for T in transforms]
    Sigmas = [np.asarray(S, float) for S in covariances]

    # Inicialización: media chordal para R + media de t
    Rm = sum(T[:3,:3] for T in Ts)/len(Ts)
    U,S,Vt = np.linalg.svd(Rm)
    R0 = U@Vt
    if np.linalg.det(R0) < 0:
        U[:,-1] *= -1
        R0 = U@Vt
    t0 = sum(T[:3,3] for T in Ts)/len(Ts)
    T_hat = np.eye(4); T_hat[:3,:3] = R0; T_hat[:3,3] = t0

    # Escala opcional m/rad
    Sscale = np.eye(6) if meters_per_radian in (None, 0.0) else np.diag(
        [1.0/(meters_per_radian)]*3 + [1,1,1]
    )

    # Información = Σ⁻¹
    Infos = []
    for Sigma in Sigmas:
        Sigma = 0.5*(Sigma + Sigma.T)
        try:
            Info = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Info = np.linalg.pinv(Sigma + 1e-9*np.eye(6))
        Infos.append(Info)

    def cost(Tcur):
        tot = 0.0
        for T, Info in zip(Ts, Infos):
            r = Sscale @ se3_log(np.linalg.inv(Tcur) @ T)
            n = np.sqrt(r.T @ (Info @ r) + 1e-16)
            if loss == 'huber':
                tot += 0.5*n*n if n <= c else c*n - 0.5*c*c
            elif loss == 'cauchy':
                z = n/c if c>0 else n
                tot += 0.5*c*c*np.log1p(z*z)
            elif loss == 'tukey':
                z = n/c if c>0 else n
                tot += (c*c/6.0)*(1-(1-z*z)**3) if abs(z) < 1 else (c*c/6.0)
            else:
                tot += 0.5*n*n
        return tot

    prev = cost(T_hat)
    H_last = np.eye(6)

    for _ in range(max_iters):
        H = np.zeros((6,6)); b = np.zeros(6)
        for T, Info in zip(Ts, Infos):
            r = Sscale @ se3_log(np.linalg.inv(T_hat) @ T)
            n = np.sqrt(r.T @ (Info @ r) + 1e-16)
            w = _robust_weight(n, c, loss)
            WI = w * Info
            H += WI
            b += WI @ r
        H += 1e-9*np.eye(6)
        try:
            delta_scaled = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            delta_scaled = np.linalg.lstsq(H, b, rcond=None)[0]
        delta = np.linalg.inv(Sscale) @ delta_scaled

        # line-search para asegurar descenso
        alpha = 1.0
        improved = False
        for _ in range(10):
            T_try = T_hat @ se3_exp(alpha*delta)
            c_try = cost(T_try)
            if c_try < prev - 1e-14:
                T_hat, prev = T_try, c_try
                improved = True
                break
            alpha *= 0.5
        if not improved and np.linalg.norm(delta) < tol:
            break
        if np.linalg.norm(delta) < tol:
            break
        H_last = H.copy()

    Cov_est = np.linalg.inv(H_last)
    return T_hat, Cov_est, prev


# ======================
#  Nodo ROS2 principal
# ======================

class ExtrinsicCalibrationOptimizer(Node):
    """
    Nodo de backend de optimización por lotes para la calibración extrínseca multi-cámara.
    Estima, para cada cámara != raíz, la transformación raíz->cámara como media robusta de múltiples mediciones.
    """
    def __init__(self):
        super().__init__('extrinsic_calibration_optimizer')

        # --- Parámetros ---
        self.declare_parameter('root_frame_id', 'cam00/camera_00')
        self.declare_parameter('camera_names', ['cam00/camera_00', 'cam00/camera_01'])
        self.declare_parameter('link_names', ['camera_00_color_optical_frame', 'camera_01_color_optical_frame'])
        self.declare_parameter('input_topic', '/calib/pairs_posecov')
        self.declare_parameter('loss', 'huber')        # 'huber'|'cauchy'|'tukey'|'none'
        self.declare_parameter('loss_c', 1.5)
        self.declare_parameter('meters_per_radian', 0.0)  # 0.0 => desactivado
        self.declare_parameter('output_dir', '/home/ubuntu/ibima-PostureSense/calib')  # Directorio donde guardar las TFs

        self.root_frame_id_ = self.get_parameter('root_frame_id').value
        self.camera_names_ = self.get_parameter('camera_names').value
        self.link_names_ = self.get_parameter('link_names').value
        self.input_topic_ = self.get_parameter('input_topic').value
        self.loss_ = self.get_parameter('loss').value
        self.loss_c_ = float(self.get_parameter('loss_c').value)
        self.mpr_ = float(self.get_parameter('meters_per_radian').value)
        self.output_dir_ = self.get_parameter('output_dir').value

        if len(self.link_names_) != len(self.camera_names_):
            self.get_logger().fatal("'camera_names' y 'link_names' deben tener la misma longitud")
            raise ValueError("camera_names/link_names mismatch")

        self.camera_name_to_link_ = {name: link for name, link in zip(self.camera_names_, self.link_names_)}

        if self.root_frame_id_ not in self.camera_names_:
            self.get_logger().fatal(f"La cámara raíz '{self.root_frame_id_}' no está en 'camera_names'.")
            raise ValueError("Cámara raíz no encontrada")

        # --- Almacenamiento de Datos ---
        self.measurement_buffer_: List[CameraPairPose] = []
        self.buffer_mutex_ = threading.Lock()

        # --- Publicador de TF Estáticas ---
        tf_static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_static_pub_ = self.create_publisher(TFMessage, '/tf_static', tf_static_qos)

        # --- Suscriptor de Mediciones ---
        self.measurement_sub_ = self.create_subscription(
            CameraPairPose, self.input_topic_, self.pair_measurement_callback, 100
        )
        self.get_logger().info(f"Suscrito a mediciones en: {self.input_topic_}")

        # --- Servidor de Optimización ---
        self.optimization_srv_ = self.create_service(
            Trigger, '~/trigger_optimization', self.optimization_service_callback
        )
        self.get_logger().info("Nodo ExtrinsicCalibrationOptimizer listo.")

    # ================
    #  Callbacks ROS2
    # ================

    def pair_measurement_callback(self, msg: CameraPairPose):
        """Acumula mediciones en el buffer de forma segura."""
        if msg.camera_from_id not in self.camera_names_ or msg.camera_to_id not in self.camera_names_:
            self.get_logger().warn(
                f"Medición con cámaras desconocidas: {msg.camera_from_id} -> {msg.camera_to_id}. Descartada."
            )
            return
        with self.buffer_mutex_:
            self.measurement_buffer_.append(msg)
        self.get_logger().info(f"Medición recibida: {msg.camera_from_id} -> {msg.camera_to_id}")

    def optimization_service_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Dispara la optimización por lotes con todas las mediciones."""
        self.get_logger().info("Llamada al servicio de optimización recibida.")
        optimized_poses_map = self.run_optimization()

        if optimized_poses_map is not None:
            self.get_logger().info("Optimización exitosa.")
            self.publish_static_transforms(optimized_poses_map)
            
            # Guardar las TFs en un archivo
            saved_file = self.save_transforms_to_file(optimized_poses_map)
            
            response.success = True
            response.message = f"Optimización completada. TFs publicadas y guardadas en: {saved_file}"
        else:
            self.get_logger().error("La optimización falló.")
            response.success = False
            response.message = "Fallo en la optimización (ver logs)."
        return response

    # ======================
    #  Lógica de optimización
    # ======================

    def _msg_to_T_Sigma(self, m: CameraPairPose) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extrae (T, Sigma) de un CameraPairPose admitiendo variantes:
        - PoseWithCovariance en m.pose
        - Transform + covariance en m.transform / m.covariance
        Devuelve None si falta información.
        """
        if hasattr(m, 'measured_transform') and isinstance(m.measured_transform, PoseWithCovariance):
                p = m.measured_transform.pose.position
                q = m.measured_transform.pose.orientation
                R = quat_to_rotmat(q.x, q.y, q.z, q.w)
                t = np.array([p.x, p.y, p.z], dtype=float)
                T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
                cov = np.array(m.measured_transform.covariance, dtype=float).reshape(6,6)
                cov = 0.5*(cov + cov.T)
                return T, cov

        self.get_logger().warn("No se pudo extraer (T,Σ) de la medición; comprueba el mensaje CameraPairPose.")
        return None

    def run_optimization(self) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Construye y resuelve el problema de optimización para cada cámara ≠ raíz.
        Devuelve: dict camera_name -> (T_hat 4x4 de root->camera, Cov_hat 6x6)
        """
        with self.buffer_mutex_:
            msgs = list(self.measurement_buffer_)
        if len(msgs) == 0:
            self.get_logger().warn("No hay mediciones en el buffer.")
            return None

        root = self.root_frame_id_
        results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for cam in self.camera_names_:
            if cam == root:
                continue

            Ts: List[np.ndarray] = []
            Sigmas: List[np.ndarray] = []

            # Recolecta todas las mediciones root<->cam (ambas direcciones)
            for m in msgs:
                if (m.camera_from_id == root and m.camera_to_id == cam) or \
                   (m.camera_from_id == cam and m.camera_to_id == root):

                    parsed = self._msg_to_T_Sigma(m)
                    if parsed is None:
                        continue
                    T_meas, Sigma_meas = parsed

                    if m.camera_from_id == root and m.camera_to_id == cam:
                        # ya está como root->cam
                        Ts.append(T_meas)
                        Sigmas.append(Sigma_meas)
                    else:
                        # viene como cam->root: invertimos y transportamos covarianza
                        T_inv = np.linalg.inv(T_meas)  # root->cam
                        Ad = adjoint_SE3(T_inv)
                        Sigma_inv = Ad @ Sigma_meas @ Ad.T
                        Sigma_inv = 0.5*(Sigma_inv + Sigma_inv.T)
                        Ts.append(T_inv)
                        Sigmas.append(Sigma_inv)

            if len(Ts) == 0:
                self.get_logger().warn(f"Sin mediciones para par {root} -> {cam}.")
                continue

            try:
                T_hat, Cov_hat, J = se3_average(
                    Ts, Sigmas,
                    loss=self.loss_, c=self.loss_c_,
                    max_iters=80, tol=1e-10,
                    meters_per_radian=(self.mpr_ if self.mpr_ > 0 else None)
                )
                results[cam] = (T_hat, Cov_hat)
                e = se3_log(np.linalg.inv(T_hat) @ Ts[0])  # error vs primera med. (solo para log)
                self.get_logger().info(
                    f"[{root}->{cam}] {len(Ts)} meas | coste={J:.6f} | "
                    f"||err_ref||={np.linalg.norm(e):.3e}"
                )
            except Exception as ex:
                self.get_logger().error(f"Fallo optimizando {root}->{cam}: {ex}")

        if len(results) == 0:
            return None
        return results

    # ==========================
    #  Publicación de /tf_static
    # ==========================

    def publish_static_transforms(self, optimized_poses_map: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Publica las poses optimizadas (relativas a la raíz) como TFs estáticas.
        """
        tf_msg = TFMessage()
        now = self.get_clock().now().to_msg()

        root_link = self.camera_name_to_link_.get(self.root_frame_id_, self.root_frame_id_)

        for cam, (T_hat, Cov_hat) in optimized_poses_map.items():
            child_link = self.camera_name_to_link_.get(cam, cam)

            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = root_link
            ts.child_frame_id = child_link

            t = T_hat[:3,3]
            qx, qy, qz, qw = rotmat_to_quat(T_hat[:3,:3])

            ts.transform.translation.x = float(t[0])
            ts.transform.translation.y = float(t[1])
            ts.transform.translation.z = float(t[2])
            ts.transform.rotation.x = float(qx)
            ts.transform.rotation.y = float(qy)
            ts.transform.rotation.z = float(qz)
            ts.transform.rotation.w = float(qw)

            tf_msg.transforms.append(ts)

        if tf_msg.transforms:
            self.get_logger().info(f"Publicando {len(tf_msg.transforms)} TFs estáticas en /tf_static")
            self.tf_static_pub_.publish(tf_msg)
        else:
            self.get_logger().warn("No hay TFs para publicar.")

    def save_transforms_to_file(self, optimized_poses_map: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> str:
        """
        Guarda las transformaciones optimizadas en un archivo JSON.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_transforms_{timestamp}.json"
        filepath = os.path.join(self.output_dir_, filename)
        
        # Crear el directorio si no existe
        os.makedirs(self.output_dir_, exist_ok=True)
        
        root_link = self.camera_name_to_link_.get(self.root_frame_id_, self.root_frame_id_)
        
        data = {
            "timestamp": timestamp,
            "root_frame": root_link,
            "transforms": []
        }
        
        for cam, (T_hat, Cov_hat) in optimized_poses_map.items():
            child_link = self.camera_name_to_link_.get(cam, cam)
            
            t = T_hat[:3, 3]
            qx, qy, qz, qw = rotmat_to_quat(T_hat[:3, :3])
            
            transform_data = {
                "child_frame": child_link,
                "camera_name": cam,
                "translation": {
                    "x": float(t[0]),
                    "y": float(t[1]),
                    "z": float(t[2])
                },
                "rotation": {
                    "x": float(qx),
                    "y": float(qy),
                    "z": float(qz),
                    "w": float(qw)
                },
                "covariance": Cov_hat.flatten().tolist(),
                "transform_matrix": T_hat.tolist()
            }
            data["transforms"].append(transform_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.get_logger().info(f"Transformaciones guardadas en: {filepath}")
        return filepath

# --- Punto de Entrada ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ExtrinsicCalibrationOptimizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().fatal(f"Error crítico: {e}")
        else:
            print(f"Error crítico al crear el nodo: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
