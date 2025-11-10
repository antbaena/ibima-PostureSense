#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nodo ROS2 para estimar T_cam->cube usando ICP point-to-plane con profundidad alineada al color.
- Sustituye el solvePnP por una optimización Gauss-Newton/LM sobre residuos punto-a-plano.
- Multi-marcador: apila una rejilla de puntos por marcador.
- Ponderado por incertidumbre de profundidad + robusta Huber.
- Soporta fallback a PnP si falta profundidad o no converge.
- Publica pares T_i_j y covarianza como en el nodo original.

Esta versión añade **logging DEBUG exhaustivo** y pequeñas comprobaciones defensivas para poder
trazar problemas de convergencia/degeneración del ICP.

Cómo ver los logs:
- Ejecuta con: `--ros-args --log-level synchronized_pair_publisher_icp:=DEBUG` o activa el parámetro `force_debug_logging`.
- Para volcado a fichero: `ros2 run ... 2>&1 | tee icp_trace.log`

"""
import rclpy
import numpy as np
import yaml
import copy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.logging import LoggingSeverity, set_logger_level

# Mensajes
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Transform, PoseWithCovariance, PoseWithCovarianceStamped, TransformStamped
from multicam_cube_calib_interfaces.msg import CameraPairPose

# OpenCV / NumPy
from cv_bridge import CvBridge
import cv2

# Sincronización
import message_filters
import tf2_ros

# --- Utilidades SE(3) ---
def _skew(v):
    v = np.asarray(v).reshape(3)
    return np.array([[0,     -v[2],  v[1]],
                     [v[2],   0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=float)

def se3_exp(xi):
    """xi = [wx, wy, wz, vx, vy, vz] -> T in SE(3)."""
    w = np.array(xi[:3], dtype=float)
    v = np.array(xi[3:], dtype=float)
    th = np.linalg.norm(w)
    W = _skew(w)
    I = np.eye(3)
    if th < 1e-12:
        R = I + W
        V = I + 0.5 * W
    else:
        a = np.sin(th)/th
        b = (1 - np.cos(th))/(th*th)
        R = I + a*W + b*(W@W)
        c = (th - np.sin(th))/(th**3)
        V = I + b*W + c*(W@W)
    t = V @ v
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def adjoint_SE3(T):
    """Adjunto de un SE(3) 4x4. Orden de twist: [wx, wy, wz, vx, vy, vz]."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:, :3] = _skew(t) @ R
    Ad[3:, 3:] = R
    return Ad

# Reordenado covarianzas
def cov_perm(cov6, frm, to):
    map_idx = {k:i for i,k in enumerate(frm)}
    idx = [map_idx[k] for k in to]
    P = np.eye(6)[idx, :]
    return P @ cov6 @ P.T

def cov_relative_right_invariant(Cov_i, Cov_j, T_j_cube, Cov_cross=None):
    AdY = adjoint_SE3(T_j_cube)                 # 6x6, orden [ω; v]
    Sigma = AdY @ Cov_i @ AdY.T + Cov_j
    if Cov_cross is not None:
        Sigma = Sigma - AdY @ Cov_cross - Cov_cross.T @ AdY.T
    return Sigma

# Rot matriz -> Pose

def mat_to_pose(matrix: np.ndarray) -> Pose:
    from scipy.spatial.transform import Rotation as R
    pose_msg = Pose()
    pose_msg.position.x = float(matrix[0, 3])
    pose_msg.position.y = float(matrix[1, 3])
    pose_msg.position.z = float(matrix[2, 3])
    r = R.from_matrix(matrix[:3, :3])
    q = r.as_quat() # [x, y, z, w]
    pose_msg.orientation.x = float(q[0])
    pose_msg.orientation.y = float(q[1])
    pose_msg.orientation.z = float(q[2])
    pose_msg.orientation.w = float(q[3])
    return pose_msg

# --- Utils debug ---

def _fmt_T(T: np.ndarray) -> str:
    try:
        from scipy.spatial.transform import Rotation as R
        t = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
        return f"t[m]={t.round(4).tolist()}, rpy[deg]={rpy.round(2).tolist()}"
    except Exception:
        t = T[:3, 3]
        return f"t[m]={t.round(4).tolist()}"


def _stats(name: str, arr: np.ndarray) -> str:
    a = np.asarray(arr).ravel()
    finite = np.isfinite(a)
    if finite.sum() == 0:
        return f"{name}: no finites (N={a.size})"
    a = a[finite]
    return (f"{name}: N={a.size}, min={np.min(a):.6g}, max={np.max(a):.6g}, "
            f"mean={np.mean(a):.6g}, std={np.std(a):.6g}")


# --- ICP auxiliar ---

def huber_weights(r, delta):
    """Devuelve pesos robustos w (multiplicativos) para residuos r (1D array)."""
    r = np.asarray(r, dtype=float).reshape(-1)
    a = np.abs(r)
    w = np.ones_like(a)
    mask = a > delta
    # Derivada de Huber -> peso equivalente delta/|r|
    w[mask] = (delta / (a[mask] + 1e-12))
    return w

class SynchronizedPairPublisherICP(Node):
    def __init__(self):
        super().__init__('synchronized_pair_publisher_icp')

        # ====== Parámetros ======
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01'])
        self.declare_parameter('image_topic_tpl', '/{}/color/image_raw/decompressed')
        # Ajusta según tu pipeline de RealSense/Orbbec/etc.
        self.declare_parameter('depth_topic_tpl', '/{}/depth/image_raw/decompressed')
        self.declare_parameter('output_topic', '/calib/pairs_posecov')

        self.declare_parameter('cube_config', '/home/ubuntu/ibima-PostureSense/code/src/multicam_cube_calib/config/cube_config_T.yaml')
        self.declare_parameter('marker_length', 0.23)
        self.declare_parameter('dictionary', 'DICT_5X5_100')

        self.declare_parameter('sync_slop_sec', 0.03) # 30ms

        # ICP/Depth params
        self.declare_parameter('depth_scale', 0.001) # mm->m
        self.declare_parameter('z_min', 0.2)
        self.declare_parameter('z_max', 5.0)
        self.declare_parameter('grid_n', 15)
        self.declare_parameter('grid_margin', 0.12) # en [-1,1] coords
        self.declare_parameter('huber_delta', 0.01) # m
        self.declare_parameter('depth_sigma_a', 0.001) # m
        self.declare_parameter('depth_sigma_b', 0.001) # m/m^2
        self.declare_parameter('max_icp_iters', 10)
        self.declare_parameter('lm_damping', 1e-6)
        self.declare_parameter('min_update_norm', 1e-6)
        self.declare_parameter('pose_init_from_prev', True)

        # Covarianza base (para fallback/priors)
        self.declare_parameter('base_variance_m', 0.001)
        self.declare_parameter('base_variance_rad', 0.002)
        self.declare_parameter('depth_variance_factor', 5.0)

        # Debug
        self.declare_parameter('debug_pose_covariance', True)
        self.declare_parameter('debug_output_topic', '/calib/debug/pose_covariance')
        self.declare_parameter('force_debug_logging', False)
        self.declare_parameter('profile_timing', True)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ====== Leer parámetros ======
        self.cameras = self.get_parameter('cameras').get_parameter_value().string_array_value
        self.image_topic_tpl = self.get_parameter('image_topic_tpl').get_parameter_value().string_value
        self.depth_topic_tpl = self.get_parameter('depth_topic_tpl').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.cube_config_path = self.get_parameter('cube_config').value
        self.marker_length = self.get_parameter('marker_length').get_parameter_value().double_value
        self.dict_name = self.get_parameter('dictionary').get_parameter_value().string_value
        self.sync_slop_sec = self.get_parameter('sync_slop_sec').get_parameter_value().double_value

        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.z_min = self.get_parameter('z_min').get_parameter_value().double_value
        self.z_max = self.get_parameter('z_max').get_parameter_value().double_value
        self.grid_n = int(self.get_parameter('grid_n').get_parameter_value().integer_value)
        self.grid_margin = self.get_parameter('grid_margin').get_parameter_value().double_value
        self.huber_delta = self.get_parameter('huber_delta').get_parameter_value().double_value
        self.depth_sigma_a = self.get_parameter('depth_sigma_a').get_parameter_value().double_value
        self.depth_sigma_b = self.get_parameter('depth_sigma_b').get_parameter_value().double_value
        self.max_icp_iters = int(self.get_parameter('max_icp_iters').get_parameter_value().integer_value)
        self.lm_damping = self.get_parameter('lm_damping').get_parameter_value().double_value
        self.min_update_norm = self.get_parameter('min_update_norm').get_parameter_value().double_value
        self.pose_init_from_prev = self.get_parameter('pose_init_from_prev').get_parameter_value().bool_value

        self.base_var_m = self.get_parameter('base_variance_m').get_parameter_value().double_value
        self.base_var_rad = self.get_parameter('base_variance_rad').get_parameter_value().double_value
        self.depth_var_factor = self.get_parameter('depth_variance_factor').get_parameter_value().double_value

        self.debug_pose_covariance = self.get_parameter('debug_pose_covariance').get_parameter_value().bool_value
        self.debug_output_topic = self.get_parameter('debug_output_topic').get_parameter_value().string_value
        self.force_debug_logging = self.get_parameter('force_debug_logging').get_parameter_value().bool_value
        self.profile_timing = self.get_parameter('profile_timing').get_parameter_value().bool_value

        if self.force_debug_logging:
            try:
                set_logger_level(self.get_logger().name, LoggingSeverity.DEBUG)
            except Exception:
                pass

        # ====== QoS / Utilidades ======
        self.bridge = CvBridge()
        self.qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.qos_images = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Cargar diccionario ArUco
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.get_logger().debug(f"Aruco dict='{self.dict_name}' cargado")
        except AttributeError:
            self.get_logger().error(f"Diccionario ArUco inválido: {self.dict_name}. Usando DICT_5X5_100.")
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Cargar configuración del cubo
        self.marker_to_cube_tf = self.load_cube_config()
        if not self.marker_to_cube_tf:
            self.get_logger().fatal("No se pudo cargar la configuración del cubo. Abortando.")
            return

        # Puntos 3D de las esquinas del marcador (en su propio frame, Z=0)
        L = self.marker_length / 2.0
        self.objp_marker_local = np.array([[-L, -L, 0], [ L, -L, 0], [ L,  L, 0], [-L,  L, 0]], dtype=np.float32)

        # ====== Estado ======
        self.infos = {}      # cam -> (K, D, (w,h))
        self.info_subs = {}  # cam -> Subscription
        self.sync_started = False
        self.synchronizer = None
        self.prev_poses = {} # cam -> T_cam_cube (4x4)

        # ====== Publishers ======
        self.pub = self.create_publisher(CameraPairPose, self.output_topic, self.qos_reliable)
        self.get_logger().info(f"Publicando pares en {self.output_topic}")

        if self.debug_pose_covariance:
            self.debug_pub = self.create_publisher(PoseWithCovarianceStamped, self.debug_output_topic, self.qos_reliable)
            self.get_logger().info(f"Publicando debug de pose con covarianza en {self.debug_output_topic}")

        # ====== Iniciar suscripciones de CameraInfo ======
        self.get_logger().info("Esperando CameraInfo de todas las cámaras...")
        for cam in self.cameras:
            info_topic = self.caminfo_topic_tpl(cam)
            sub = self.create_subscription(
                CameraInfo,
                info_topic,
                lambda msg, c=cam: self.on_info(c, msg),
                self.qos_images
            )
            self.info_subs[cam] = sub
            self.get_logger().debug(f"Suscrito a CameraInfo: {info_topic}")

    # Helpers de topics
    def caminfo_topic_tpl(self, cam):
        return '/{}/color/camera_info'.format(cam)

    def color_topic_tpl(self, cam):
        return self.image_topic_tpl.format(cam)

    def depth_topic_tpl_f(self, cam):
        return self.depth_topic_tpl.format(cam)

    def load_cube_config(self) -> dict:
        marker_poses = {}
        try:
            with open(self.cube_config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            for mid, v in (cfg.get('markers', {}) or {}).items():
                try:
                    mid_i = int(mid)
                    T = np.array(v['T'], dtype=float).reshape(4, 4)
                    marker_poses[mid_i] = T
                except Exception:
                    self.get_logger().warning(f"Formato inválido para marker {mid} en config; ignorando")
            self.get_logger().info(f"Cargadas {len(marker_poses)} poses de marcadores desde {self.cube_config_path}")
            return marker_poses
        except Exception as e:
            self.get_logger().error(f"No se pudo cargar cube_config '{self.cube_config_path}': {e}")
            return {}

    def on_info(self, cam: str, msg: CameraInfo):
        if cam in self.infos:
            return
        try:
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            D = np.array(msg.d, dtype=float)
        except Exception:
            self.get_logger().warning(f"CameraInfo inválido para {cam}; ignorando")
            return
        if msg.width == 0 or np.allclose(K, 0):
            self.get_logger().info(f"CameraInfo incompleto para {cam}; esperando...")
            return
        self.get_logger().info(f"CameraInfo recibido para {cam}. {K.shape=} {len(D)=} {msg.width=}x{msg.height}")
        self.infos[cam] = (K, D, (msg.width, msg.height))
        sub = self.info_subs.pop(cam, None)
        if sub:
            self.destroy_subscription(sub)
        self.check_and_start_synchronizer()

    def check_and_start_synchronizer(self):
        if self.sync_started or len(self.infos) != len(self.cameras):
            return
        self.sync_started = True
        self.get_logger().info("¡Todas las CameraInfos recibidas! Iniciando sincronizador de imágenes RGBD.")

        subs = []
        for cam in self.cameras:
            c_topic = self.color_topic_tpl(cam)
            d_topic = self.depth_topic_tpl_f(cam)
            self.get_logger().info(f"Suscrito a COLOR (sync): {c_topic}")
            self.get_logger().info(f"Suscrito a DEPTH (sync): {d_topic}")
            subs.append(message_filters.Subscriber(self, Image, c_topic, qos_profile=self.qos_images))
            subs.append(message_filters.Subscriber(self, Image, d_topic, qos_profile=self.qos_images))

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            subs,
            queue_size=10,
            slop=self.sync_slop_sec
        )
        self.synchronizer.registerCallback(self.synchronized_callback)

    def synchronized_callback(self, *msgs):
        """Recibe [color1, depth1, color2, depth2, ...] sincronizados."""
        t_cb0 = time.time() if self.profile_timing else None
        n = len(self.cameras)
        if len(msgs) != 2*n:
            self.get_logger().warning("Callback RGBD con número inesperado de mensajes.")
            return

        # Usamos el stamp de la primera imagen como trace id
        h0 = msgs[0].header
        trace_id = f"{h0.stamp.sec}.{h0.stamp.nanosec:09d}"
        self.get_logger().debug(f"[TRACE {trace_id}] callback con {n} cámaras")

        poses_with_cov = {}

        for idx, cam in enumerate(self.cameras):
            img_msg = msgs[2*idx]
            dep_msg = msgs[2*idx + 1]
            K, D, _ = self.infos[cam]
            self.get_logger().debug(
                f"[TRACE {trace_id}] [{cam}] enc_color='{img_msg.encoding}' enc_depth='{dep_msg.encoding}' size={img_msg.width}x{img_msg.height}")

            # Convertir imágenes
            try:
                gray = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] Imagen color->gray OK")
            except Exception:
                gray = self.bridge.imgmsg_to_cv2(img_msg)
                if gray.ndim == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] Imagen color->gray por fallback")

            depth = self.bridge.imgmsg_to_cv2(dep_msg)
            d_m = self.depth_to_meters(depth, cam, trace_id)

            # Estimar pose por ICP
            T_init = self.prev_poses.get(cam)
            if T_init is not None:
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] init=prev_pose {_fmt_T(T_init)}")
            else:
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] init=NONE (intentará PnP)")

            t0 = time.time()
            T_cam_cube, Cov = self.estimate_pose_icp(gray, d_m, K, D, T_init)
            if self.profile_timing:
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] ICP tiempo={1000*(time.time()-t0):.1f} ms")

            if T_cam_cube is None:
                # Fallback a PnP si falla ICP
                self.get_logger().warning(f"[TRACE {trace_id}] [{cam}] ICP falló; intentando PnP como fallback.")
                t1 = time.time()
                T_cam_cube, Cov = self.estimate_pose_pnp(gray, K, D)
                if self.profile_timing:
                    self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] PnP tiempo={1000*(time.time()-t1):.1f} ms")

            if T_cam_cube is not None:
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] Pose OK {_fmt_T(T_cam_cube)}")
                poses_with_cov[cam] = (T_cam_cube, Cov)
                self.prev_poses[cam] = T_cam_cube
            else:
                self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] No se pudo estimar pose (ni ICP ni PnP)")

        if len(poses_with_cov) < 2:
            self.get_logger().debug(f"[TRACE {trace_id}] Menos de 2 cámaras vieron/estimaron el cubo.")
            return

        cams = list(poses_with_cov.keys())
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                ci, cj = cams[i], cams[j]
                Ti, Covi = poses_with_cov[ci]
                Tj, Covj = poses_with_cov[cj]
                try:
                    Tij = Ti @ np.linalg.inv(Tj)
                except np.linalg.LinAlgError:
                    self.get_logger().warning(f"[TRACE {trace_id}] Error al invertir T de {cj}")
                    continue
                self.get_logger().debug(f"[TRACE {trace_id}] Pair {ci}->{cj} {_fmt_T(Tij)}")
                Covi_twist = cov_perm(Covi, ['x','y','z','roll','pitch','yaw'], ['roll','pitch','yaw','x','y','z'])
                Covj_twist = cov_perm(Covj, ['x','y','z','roll','pitch','yaw'], ['roll','pitch','yaw','x','y','z'])
                Cov_ij_twist = cov_relative_right_invariant(Covi_twist, Covj_twist, Tj)
                Cov_ij = cov_perm(Cov_ij_twist, ['roll','pitch','yaw','x','y','z'], ['x','y','z','roll','pitch','yaw'])
                self.publish_pair_pose(ci, cj, Tij, Cov_ij, msgs[0].header)

        if self.profile_timing:
            self.get_logger().debug(f"[TRACE {trace_id}] callback total {1000*(time.time()-t_cb0):.1f} ms")

    # --- Conversión depth a metros ---
    def depth_to_meters(self, depth, cam: str = "?", trace_id: str = "-"):
        if depth is None:
            self.get_logger().debug(f"[TRACE {trace_id}] [{cam}] depth=None")
            return None
        try:
            if depth.dtype == np.uint16:
                d = depth.astype(np.float32) * self.depth_scale
                enc = '16UC1'
            elif depth.dtype in (np.float32, np.float64):
                d = depth.astype(np.float32)
                enc = '32FC1'
            else:
                d = depth.astype(np.float32)
                enc = str(depth.dtype)
            # Marca como NaN los ceros/negativos
            bad = (d <= 0.0)
            d[bad] = np.nan
            self.get_logger().debug(
                f"[TRACE {trace_id}] [{cam}] depth enc={enc} scale={self.depth_scale} "
                f"{_stats('depth[m]', d)} NaNs={(~np.isfinite(d)).sum()}")
            return d
        except Exception as e:
            self.get_logger().warning(f"[TRACE {trace_id}] [{cam}] Error en depth_to_meters: {e}")
            return None

    # --- Estimadores de pose ---
    def estimate_pose_pnp(self, gray, K, D):
        t0 = time.time()
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            self.get_logger().debug("PnP: 0 marcadores detectados")
            return None, None
        self.get_logger().debug(f"PnP: detectados {len(ids)} marcadores: {ids.ravel().tolist()}")
        all_object_points = []
        all_image_points = []
        used_ids = []
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            T_marker_to_cube = self.marker_to_cube_tf.get(int(marker_id))
            if T_marker_to_cube is None:
                self.get_logger().debug(f"PnP: marker {int(marker_id)} no está en config -> omitido")
                continue
            objp_marker_h = np.hstack([self.objp_marker_local, np.ones((4, 1))])
            objp_cube_h = (T_marker_to_cube @ objp_marker_h.T).T
            all_object_points.append(objp_cube_h[:, :3])
            all_image_points.append(marker_corners.reshape(-1, 2))
            used_ids.append(int(marker_id))
            break
        if len(all_object_points) == 0:
            self.get_logger().debug("PnP: 0 marcadores utilizables (no están en config)")
            return None, None
        final_obj = np.vstack(all_object_points).astype(np.float32)
        final_img = np.vstack(all_image_points).astype(np.float32)
        try:
            ok, rvec, tvec = cv2.solvePnP(final_obj, final_img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                self.get_logger().debug("PnP: solvePnP devolvió ok=False")
                return None, None
        except Exception as e:
            self.get_logger().warning(f"PnP: solvePnP falló: {e}")
            return None, None
        R_cam_cube, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3,:3] = R_cam_cube
        T[:3, 3] = tvec.flatten()

        # Cov heurística tipo PnP (como nodo original)
        proj_pts, _ = cv2.projectPoints(final_obj, rvec, tvec, K, D)
        reproj = np.mean(np.linalg.norm(proj_pts.reshape(-1, 2) - final_img.reshape(-1, 2), axis=1))
        num_points = len(final_obj)
        quality = float(num_points) / max(1e-3, reproj**2)
        var_scale = 1.0 / max(1e-6, quality)
        var_xy = self.base_var_m * var_scale
        var_z  = self.base_var_m * var_scale * self.depth_var_factor
        var_r  = self.base_var_rad * var_scale
        cov_matrix = np.diag([var_xy, var_xy, var_z, var_r, var_r, var_r])
        self.get_logger().debug(
            f"PnP: usados {len(used_ids)} marcadores {used_ids}, puntos={num_points}, reproj={reproj:.3g} px, "
            f"var_scale={var_scale:.3g}")
        if self.profile_timing:
            self.get_logger().debug(f"PnP: tiempo total {1000*(time.time()-t0):.1f} ms")
        return T, cov_matrix

    def estimate_pose_icp(self, gray, depth_m, K, D, T_init=None):
        t0 = time.time()
        if depth_m is None:
            self.get_logger().debug("ICP: depth=None -> abort")
            return None, None
        h, w = gray.shape[:2]
        invK = np.linalg.inv(K)

        # 1) Detectar marcadores
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            self.get_logger().debug("ICP: 0 marcadores detectados")
            return None, None
        ids_list = ids.flatten().tolist()
        self.get_logger().debug(f"ICP: detectados {len(ids_list)} marcadores: {ids_list}")
        
        # 2) Construir rejilla de muestras por marcador
        samples_X = []  # puntos en cámara 3D (se recomputan por depth, independ. de T)
        plane_ns = []   # normales n en frame del cubo (3,)
        plane_Y0 = []   # puntos Y0 (centro) en frame del cubo (3,)
        weights0 = []   # pesos por geometría/ángulo/borde (sin robusta aún)

        per_marker_counts = {}

        # Rejilla en coords normalizadas de marcador [-1,1]
        gN = max(3, int(self.grid_n))
        m = float(self.grid_margin)
        xs = np.linspace(-1+m, 1-m, gN)
        ys = np.linspace(-1+m, 1-m, gN)
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        self.get_logger().debug(f"ICP: rejilla {gN}x{gN}, margin={m}")

        # Orden de esquinas ArUco típico: TL, TR, BR, BL
        src_square = np.array([[-1,-1],
                               [1, -1],
                               [1,  1],
                               [-1, 1]], dtype=np.float32)

        for marker_id, marker_corners in zip(ids.flatten(), corners):
            T_m_c = self.marker_to_cube_tf.get(int(marker_id))
            if T_m_c is None:
                self.get_logger().debug(f"ICP: marker {int(marker_id)} no está en config -> omitido")
                continue
            dst_px = marker_corners.reshape(4,2).astype(np.float32)
            H = cv2.getPerspectiveTransform(src_square, dst_px)

            # Ideal plane (en frame del cubo)
            n_c = (T_m_c[:3,:3] @ np.array([0,0,1.0]))  # normal +Z
            Y0_c = (T_m_c @ np.array([0,0,0,1.0]))[:3]

            # Para peso por ángulo de incidencia: n en cámara depende de T actual; usa aprox inicial
            if T_init is None:
                n_cam_z = 1.0
            else:
                n_cam = T_init[:3,:3].T @ n_c
                n_cam_z = max(0.05, float(n_cam[2]))

            used_this_marker = 0
            for uvn in grid:
                # u,v por homografía
                p = np.array([uvn[0], uvn[1], 1.0], dtype=np.float32)
                q = H @ p
                if q[2] == 0:
                    continue
                u = float(q[0]/q[2])
                v = float(q[1]/q[2])
                # ventana 3x3 para mediana, evitando bordes 1px
                ui = int(round(u))
                vi = int(round(v))
                if ui < 1 or ui >= (w-1) or vi < 1 or vi >= (h-1):
                    continue
                patch = depth_m[vi-1:vi+2, ui-1:ui+2]
                z = np.nanmedian(patch)
                if not np.isfinite(z) or z < self.z_min or z > self.z_max:
                    continue
                # Back-project
                pix = np.array([u, v, 1.0], dtype=float)
                Xc = z * (invK @ pix)

                # Peso geométrico: distancia a borde (máximo en centro)
                db = min(uvn[0]-(-1+m), (1-m)-uvn[0], uvn[1]-(-1+m), (1-m)-uvn[1])
                w_border = max(1e-3, float(db))
                # Peso por ángulo (evita rasante)
                w_angle = float(n_cam_z)
                samples_X.append(Xc)
                plane_ns.append(n_c)
                plane_Y0.append(Y0_c)
                weights0.append(w_border * w_angle)
                used_this_marker += 1

            per_marker_counts[int(marker_id)] = used_this_marker

        total_samples = len(samples_X)
        self.get_logger().debug(f"ICP: muestras totales={total_samples}, por marcador={per_marker_counts}")
        if total_samples < 20:
            self.get_logger().debug("ICP: <20 muestras -> abort")
            return None, None

        X = np.asarray(samples_X, dtype=float)
        Ns = np.asarray(plane_ns, dtype=float)
        Y0s = np.asarray(plane_Y0, dtype=float)
        w0 = np.asarray(weights0, dtype=float)

        # 3) Inicialización
        if self.pose_init_from_prev and T_init is not None:
            T = T_init.copy()
            self.get_logger().debug(f"ICP: init=prev {_fmt_T(T)}")
        else:
            T, _ = self.estimate_pose_pnp(gray, K, D)
            if T is None:
                T = np.eye(4)
                self.get_logger().debug("ICP: init=IDENTITY (PnP no disponible)")
            else:
                self.get_logger().debug(f"ICP: init=PnP {_fmt_T(T)}")

        # 4) ICP point-to-plane Gauss-Newton/LM
        lam = self.lm_damping
        delta_prev = 1e9
        for it in range(self.max_icp_iters):
            R = T[:3,:3]
            t = T[:3, 3]
            RX = (R @ X.T).T  # (N,3)
            # Residuales r_k = n^T (R X + t - Y0)
            r = np.sum(Ns * (RX + t - Y0s), axis=1)

            # Jacobianos J_k = [ - n^T R [X]_x   ,  n^T ]
            Np = X.shape[0]
            J = np.zeros((Np, 6), dtype=float)
            for k in range(Np):
                n = Ns[k]
                Xk = X[k]
                RXskew = R @ _skew(Xk)
                J[k, :3] = - (n @ RXskew)   # (1x3)
                J[k, 3:] = n                 # (1x3)

            # Pesos por incertidumbre de profundidad
            z = np.linalg.norm(X, axis=1)
            sigma_z = self.depth_sigma_a + self.depth_sigma_b * (z**2)
            wz = 1.0 / (sigma_z**2 + 1e-12)

            # Robust Huber
            w_rob = huber_weights(r, self.huber_delta)

            # Pesos totales
            w = w0 * wz * w_rob
            JT_W = (J.T * w.reshape(-1))
            A = JT_W @ J + lam * np.eye(6)
            b = - JT_W @ r

            try:
                dxi = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                try:
                    cond = np.linalg.cond(A)
                except Exception:
                    cond = -1
                self.get_logger().warning(f"ICP: sistema singular en iter {it}, cond(A)={cond:.3g}; abortando.")
                return None, None

            T = se3_exp(dxi) @ T

            # Logs iteración
            upd_norm = float(np.linalg.norm(dxi))
            inliers = int(np.sum(np.abs(r) <= self.huber_delta))
            outliers = int(len(r) - inliers)
            self.get_logger().debug(
                f"ICP it={it} |dξ|={upd_norm:.3e}, inliers={inliers}, outliers={outliers}, "
                f"{_stats('r[m]', r)}")

            if upd_norm < self.min_update_norm:
                self.get_logger().debug(f"ICP: converge por |dξ|<{self.min_update_norm}")
                break
            if abs(delta_prev - upd_norm) < 1e-12:
                self.get_logger().debug("ICP: estancamiento (delta_prev≈upd_norm)")
                break
            delta_prev = upd_norm

        # 5) Covarianza Σ ≈ σ^2 (J^T W J)^{-1}
        R = T[:3,:3]
        t = T[:3, 3]
        RX = (R @ X.T).T
        r = np.sum(Ns * (RX + t - Y0s), axis=1)
        J = np.zeros((X.shape[0], 6), dtype=float)
        for k in range(X.shape[0]):
            n = Ns[k]
            Xk = X[k]
            RXskew = R @ _skew(Xk)
            J[k, :3] = - (n @ RXskew)
            J[k, 3:] = n
        z = np.linalg.norm(X, axis=1)
        sigma_z = self.depth_sigma_a + self.depth_sigma_b * (z**2)
        wz = 1.0 / (sigma_z**2 + 1e-12)
        w_rob = huber_weights(r, self.huber_delta)
        w = w0 * wz * w_rob
        JT_W = (J.T * w.reshape(-1))
        H = JT_W @ J
        try:
            H_inv = np.linalg.inv(H + 1e-9*np.eye(6))
            inv_ok = True
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)
            inv_ok = False
        dof = max(1, X.shape[0] - 6)
        sigma2 = float((w * (r**2)).sum()) / dof
        Cov_twist = sigma2 * H_inv  # orden [ω,v]
        Cov_ros = cov_perm(Cov_twist, ['roll','pitch','yaw','x','y','z'], ['x','y','z','roll','pitch','yaw'])
        self.get_logger().debug(
            f"ICP fin: {_fmt_T(T)}, sigma2={sigma2:.3e}, inv_ok={inv_ok}, "
            f"diag(Cov)={np.diag(Cov_ros).round(3).tolist()}")
        if self.profile_timing:
            self.get_logger().debug(f"ICP: tiempo total {1000*(time.time()-t0):.1f} ms")

        return T, Cov_ros

    def publish_pair_pose(self, ci: str, cj: str, Tij: np.ndarray, Cov_ij: np.ndarray, header):
        out = CameraPairPose()
        out.header.stamp = header.stamp
        out.header.frame_id = ci
        out.camera_from_id = ci
        out.camera_to_id = cj
        pwc = PoseWithCovariance()
        pwc.pose = mat_to_pose(Tij)
        pwc.covariance = Cov_ij.flatten().tolist()
        out.measured_transform = pwc
        self.pub.publish(out)
        self.get_logger().debug(
            f"Publicado par {ci}->{cj}: {_fmt_T(Tij)} | diag(Cov)={np.diag(Cov_ij).round(3).tolist()}")
        if self.debug_pose_covariance:
            debug_pose = PoseWithCovarianceStamped()
            debug_pose.header = out.header
            debug_pose.pose = pwc
            self.debug_pub.publish(debug_pose)
            # Publicar también el TF entre ci -> cj
            t = TransformStamped()
            t.header.stamp = header.stamp
            t.header.frame_id = ci
            t.child_frame_id = cj
            t.transform.translation.x = pwc.pose.position.x
            t.transform.translation.y = pwc.pose.position.y
            t.transform.translation.z = pwc.pose.position.z
            t.transform.rotation = pwc.pose.orientation
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedPairPublisherICP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().fatal(f"Error crítico en spin: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
