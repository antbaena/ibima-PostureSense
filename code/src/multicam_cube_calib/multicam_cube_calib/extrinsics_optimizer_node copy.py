#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
import threading
import numpy as np

# --- ¡CAMBIO IMPORTANTE! ---
# Importamos el mensaje real del frontend, no el placeholder
from multicam_cube_calib_interfaces.msg import CameraPairPose
from geometry_msgs.msg import PoseWithCovariance, TransformStamped, Transform
# --- Fin del Cambio ---

from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

# --- Importaciones clave de GTSAM ---
import gtsam
from gtsam.symbol_shorthand import X # Para claves (X(0), X(1), ...)

class ExtrinsicCalibrationOptimizer(Node):
    """
    Nodo de backend de optimización por lotes para la calibración extrínseca
    multi-cámara.
    """
    def __init__(self):
        super().__init__('extrinsic_calibration_optimizer')

        # --- Parámetros ---
        self.declare_parameter('root_frame_id', 'cam01/camera_02')
        self.declare_parameter('camera_names', ['cam01/camera_02', 'cam02/camera_03'])
        self.declare_parameter('link_names', ['camera_02_color_optical_frame', 'camera_03_color_optical_frame'])
        self.declare_parameter('input_topic', '/calib/pairs_posecov')
        
        self.root_frame_id_ = self.get_parameter('root_frame_id').value
        self.camera_names_ = self.get_parameter('camera_names').value
        self.link_names_ = self.get_parameter('link_names').value
        self.input_topic_ = self.get_parameter('input_topic').value

        if len(self.link_names_) != len(self.camera_names_):
            self.get_logger().fatal(
                "'camera_names' and 'link_names' deben tener la misma longitud"
            )
            raise ValueError("camera_names/link_names mismatch")

        # Mapeo de nombre de cámara -> nombre de link (frame)
        self.camera_name_to_link_ = {
            name: link for name, link in zip(self.camera_names_, self.link_names_)
        }
        self.link_names_ = self.get_parameter('link_names').value

        # --- Mapeo de Nombres a Claves de GTSAM ---
        # self.camera_name_to_key_ = {'cam_0': X(0), 'cam_1': X(1), ...}
        self.camera_name_to_key_ = {
            name: X(i) for i, name in enumerate(self.camera_names_)
        }
        self.get_logger().info(f"Mapeo de cámaras: {self.camera_name_to_key_}")
        if self.root_frame_id_ not in self.camera_name_to_key_:
            self.get_logger().fatal(
                f"La cámara raíz '{self.root_frame_id_}' no está en 'camera_names'.")
            raise ValueError("Cámara raíz no encontrada")
        
        # --- Almacenamiento de Datos ---
        self.measurement_buffer_ = []
        self.buffer_mutex_ = threading.Lock()

        # --- Publicador de TF Estáticas ---
        tf_static_qos = QoSProfile(
            depth=1, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.tf_static_pub_ = self.create_publisher(
            TFMessage, '/tf_static', tf_static_qos
        )

        # --- Suscriptor de Mediciones ---
        self.measurement_sub_ = self.create_subscription(
            CameraPairPose, # <--- ¡CAMBIO!
            self.input_topic_,  # <--- ¡CAMBIO!
            self.pair_measurement_callback,
            100 # QoS profundo para no perder mediciones
        )
        self.get_logger().info(f"Suscrito a mediciones en: {self.input_topic_}")

        # --- Servidor de Optimización ---
        self.optimization_srv_ = self.create_service(
            Trigger,
            '~/trigger_optimization',
            self.optimization_service_callback
        )
        
        self.get_logger().info("Nodo ExtrinsicCalibrationOptimizer listo.")

    def pair_measurement_callback(self, msg: CameraPairPose): # <--- ¡CAMBIO!
        """ Acumula mediciones en el buffer de forma segura. """
        if msg.camera_from_id not in self.camera_name_to_key_ or msg.camera_to_id not in self.camera_name_to_key_:
            self.get_logger().warn(
                f"Medición recibida con cámaras desconocidas: "
                f"{msg.camera_from_id}, {msg.camera_to_id}. Descartada."
            )
            return
        with self.buffer_mutex_:
            self.measurement_buffer_.append(msg)
        self.get_logger().info(
            f"Medición recibida: {msg.camera_from_id} -> {msg.camera_to_id}"
        )

    def optimization_service_callback(self, 
                                    request: Trigger.Request, 
                                    response: Trigger.Response):
        """ Dispara la optimización por lotes con todas las mediciones. """
        
        self.get_logger().info("Llamada al servicio de optimización recibida.")
        
        # --- ¡CAMBIO! Lógica implementada ---
        optimized_poses_map = self.run_optimization() 

        if optimized_poses_map is not None:
            self.get_logger().info("Optimización exitosa.")
            self.publish_static_transforms(optimized_poses_map)
            response.success = True
            response.message = f"Optimización completada. TFs publicadas."
        else:
            self.get_logger().error("La optimización falló.")
            response.success = False
            response.message = "Fallo en la optimización (ver logs)."
        
        return response

    # --- ¡CAMBIO! Lógica implementada ---
    def run_optimization(self) -> dict | None:
        """
        Función principal: Construye y resuelve el problema de 
        optimización de grafos con GTSAM.
        """
        
        # 1. Copiar las mediciones de forma segura
        with self.buffer_mutex_:
            if not self.measurement_buffer_:
                self.get_logger().warn("No hay mediciones en el buffer. Abortando.")
                return None
            measurements = list(self.measurement_buffer_)
        
        self.get_logger().info(f"Iniciando optimización con {len(measurements)} mediciones.")

        # 2. Crear el Grafo y las Estimaciones Iniciales
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # 3. Añadir el Factor Prior (Fijar la Raíz)
        root_key = self.camera_name_to_key_[self.root_frame_id_]
        root_pose = gtsam.Pose3() # Identidad
        
        # Covarianza muy pequeña (alta información) para fijar el nodo
        # Orden: [roll, pitch, yaw, x, y, z] en GTSAM
        prior_cov = np.diag([1e-9] * 6) 
        # ¡OJO! GTSAM usa orden [roll, pitch, yaw, x, y, z]
        # Nuestro nodo frontend usa [x, y, z, roll, pitch, yaw]
        # Debemos reordenar la matriz de covarianza del prior.
        prior_noise_model_gtsam_order = gtsam.noiseModel.Gaussian.Covariance(prior_cov)
        graph.add(gtsam.PriorFactorPose3(root_key, root_pose, prior_noise_model_gtsam_order))

        # 4. Añadir Estimaciones Iniciales para TODOS los nodos
        for name, key in self.camera_name_to_key_.items():
            initial_estimates.insert(key, gtsam.Pose3()) 

        # 5. Añadir Factores de Medición (Aristas Ponderadas)
        num_factors = 0
        for msg in measurements:
            try:
                key_from = self.camera_name_to_key_[msg.camera_from_id]
                key_to = self.camera_name_to_key_[msg.camera_to_id]
            except KeyError as e:
                self.get_logger().warn(f"Medición descartada: cámara desconocida {e}")
                continue

            # Convertir la medición de ROS a GTSAM
            # msg.measured_transform es un PoseWithCovariance
            T_from_to_gtsam = ros_pose_to_gtsam_pose3(msg.measured_transform)
            
            # *** La Ponderación (Clave) ***
            cov_matrix_ros_order = ros_covariance_to_numpy(msg.measured_transform.covariance)
            
            # ¡OJO! Reordenar la covarianza de ROS [x,y,z,r,p,y] a GTSAM [r,p,y,x,y,z]
            cov_matrix_gtsam_order = reorder_covariance_ros_to_gtsam(cov_matrix_ros_order)
            
            # Crear el modelo de ruido de GTSAM desde la covarianza
            # Añadir 'jitter' para estabilidad numérica si la covarianza es 0
            cov_matrix_gtsam_order += np.diag([1e-9] * 6)
            noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_matrix_gtsam_order)
            
            graph.add(gtsam.BetweenFactorPose3(
                key_from, key_to, T_from_to_gtsam, noise_model
            ))
            num_factors += 1

        if num_factors == 0:
            self.get_logger().error("No se añadieron factores válidos al grafo.")
            return None

        # 6. Configurar y Ejecutar el Optimizador
        self.get_logger().info("Iniciando optimizador Levenberg-Marquardt...")
        try:
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
            result_values = optimizer.optimize()
        except Exception as e:
            self.get_logger().fatal(f"Fallo en la optimización de GTSAM: {e}")
            return None

        self.get_logger().info(
            f"Optimización finalizada. "
            f"Error inicial: {graph.error(initial_estimates):.4f}, "
            f"Error final: {graph.error(result_values):.4f}"
        )

        # 7. Extraer Resultados (Relativos a la Raíz)
        optimized_poses_map = {}
        T_world_to_root = result_values.atPose3(root_key)
        T_root_to_world = T_world_to_root.inverse()

        for name, key in self.camera_name_to_key_.items():
            T_world_to_camX = result_values.atPose3(key)
            T_root_to_camX = T_root_to_world.compose(T_world_to_camX)
            optimized_poses_map[name] = T_root_to_camX
            
        return optimized_poses_map

    # --- ¡CAMBIO! Lógica implementada ---
    def publish_static_transforms(self, optimized_poses_map: dict):
        """
        Publica las poses optimizadas (relativas a la raíz) 
        como TFs estáticas.
        """
        tf_msg = TFMessage()
        now = self.get_clock().now().to_msg()
        
        for cam_name, gtsam_pose in optimized_poses_map.items():

            if cam_name == self.root_frame_id_:
                continue
            # Obtener el nombre de link correspondiente a la cámara
            try:
                child_frame = self.camera_name_to_link_[cam_name]
            except KeyError:
                # Si no existe el link, usar el nombre de cámara como fallback
                self.get_logger().warn(
                    f"No se encontró link para '{cam_name}', usando nombre de cámara como child_frame_id"
                )
                child_frame = cam_name

            # Usar el link de la cámara raíz como frame_id (si existe)
            if self.root_frame_id_ in self.camera_name_to_link_:
                header_frame = self.camera_name_to_link_[self.root_frame_id_]
            else:
                header_frame = self.root_frame_id_

            tfs = TransformStamped()
            tfs.header.stamp = now
            tfs.header.frame_id = header_frame
            tfs.child_frame_id = child_frame
            tfs.transform = gtsam_pose3_to_ros_transform(gtsam_pose)
            
            tf_msg.transforms.append(tfs)

        if tf_msg.transforms:
            self.get_logger().info(
                f"Publicando {len(tf_msg.transforms)} TFs estáticas en /tf_static"
            )
            self.tf_static_pub_.publish(tf_msg)

# --- Funciones Helper (fuera de la clase) ---

def ros_pose_to_gtsam_pose3(ros_pose_cov: PoseWithCovariance) -> gtsam.Pose3:
    """ Convierte un Pose de ROS a un gtsam.Pose3. """
    p = ros_pose_cov.pose.position
    q = ros_pose_cov.pose.orientation
    return gtsam.Pose3(
        gtsam.Rot3(q.w, q.x, q.y, q.z), 
        gtsam.Point3(p.x, p.y, p.z)
    )

def gtsam_pose3_to_ros_transform(gtsam_pose: gtsam.Pose3) -> Transform:
    """ Convierte un gtsam.Pose3 a un Transform de ROS. """
    t = gtsam_pose.translation()
    q = gtsam_pose.rotation().toQuaternion()
    
    ros_transform = Transform()
    ros_transform.translation.x = t[0]
    ros_transform.translation.y = t[1]
    ros_transform.translation.z = t[2]
    ros_transform.rotation.x = q.x()
    ros_transform.rotation.y = q.y()
    ros_transform.rotation.z = q.z()
    ros_transform.rotation.w = q.w()
    return ros_transform

def ros_covariance_to_numpy(ros_cov_array: np.ndarray) -> np.ndarray:
    """ Convierte el array plano de covarianza de ROS a una matriz 6x6. """
    return np.array(ros_cov_array).reshape(6, 6)

def reorder_covariance_ros_to_gtsam(ros_cov: np.ndarray) -> np.ndarray:
    """
    Reordena una matriz de covarianza 6x6 del orden ROS al orden GTSAM.
    ROS:   [x, y, z, roll, pitch, yaw] (índices 0, 1, 2, 3, 4, 5)
    GTSAM: [roll, pitch, yaw, x, y, z] (índices 3, 4, 5, 0, 1, 2)
    """
    gtsam_indices = [3, 4, 5, 0, 1, 2]
    # np.ix_ crea un "indexador" de malla abierta
    # Esto reordena tanto las filas como las columnas
    return ros_cov[np.ix_(gtsam_indices, gtsam_indices)]

# --- Punto de Entrada ---
def main(args=None):
    rclpy.init(args=args)
    try:
        node = ExtrinsicCalibrationOptimizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().fatal(f"Error crítico: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()