#!/usr/bin/env python3
import rclpy
import numpy as np
import yaml
import copy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Mensajes
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Transform, PoseWithCovariance, PoseWithCovarianceStamped
# Este es el mensaje de salida que tu 'PairPosePublisher' ya usaba
from multicam_cube_calib_interfaces.msg import CameraPairPose 

# OpenCV / NumPy
from cv_bridge import CvBridge
import cv2

# Sincronización
import message_filters
import numpy as np
from sensor_msgs.msg import Imu

def _skew(v):
    return np.array([[0,     -v[2],  v[1]],
                     [v[2],   0,    -v[0]],
                     [-v[1],  v[0],  0   ]], dtype=float)

def adjoint_SE3(T):
    """Adjunto de un SE(3) 4x4. Orden de twist: [wx, wy, wz, vx, vy, vz]."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:, :3] = _skew(t) @ R
    Ad[3:, 3:] = R
    return Ad

def cov_perm(cov6, frm, to):
    """
    Reordena una covarianza 6x6 entre órdenes de ejes.
    frm/to son listas con alguna de estas etiquetas: 'x','y','z','roll','pitch','yaw'
    """
    map_idx = {k:i for i,k in enumerate(frm)}
    idx = [map_idx[k] for k in to]
    P = np.eye(6)[idx, :]
    return P @ cov6 @ P.T

def cov_relative_right_invariant(Cov_i, Cov_j, T_j_cube, Cov_cross=None):
    """
    Propaga Σ_Z de Z = X Y^{-1} con ruido right-invariant.
    Entrada/salida en orden de twist [roll, pitch, yaw, x, y, z] (ω luego v).
    Si tus Cov_i/ Cov_j están en orden ROS [x,y,z,roll,pitch,yaw], reordénalas antes.
    """
    AdY = adjoint_SE3(T_j_cube)                 # 6x6, orden [ω; v]
    Sigma = AdY @ Cov_i @ AdY.T + Cov_j
    if Cov_cross is not None:
        Sigma = Sigma - AdY @ Cov_cross - Cov_cross.T @ AdY.T
    return Sigma

# Utilidades (puedes moverlas a multicam_cube_calib.se3)
def mat_to_pose(matrix: np.ndarray) -> Pose:
    """Convierte una matriz 4x4 a geometry_msgs/Pose"""
    from scipy.spatial.transform import Rotation as R
    pose_msg = Pose()
    
    # Traslación
    pose_msg.position.x = matrix[0, 3]
    pose_msg.position.y = matrix[1, 3]
    pose_msg.position.z = matrix[2, 3]
    
    # Rotación (usando scipy para convertir de matriz a cuaternión)
    r = R.from_matrix(matrix[:3, :3])
    q = r.as_quat() # [x, y, z, w]
    pose_msg.orientation.x = q[0]
    pose_msg.orientation.y = q[1]
    pose_msg.orientation.z = q[2]
    pose_msg.orientation.w = q[3]
    return pose_msg

class SynchronizedPairPublisher(Node):
    def __init__(self):
        super().__init__('synchronized_pair_publisher')

        # ====== Parámetros ======
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01'])
        self.declare_parameter('image_topic_tpl', '/{}/color/image_raw/decompressed')
        self.declare_parameter('caminfo_topic_tpl', '/{}/color/camera_info')
        self.declare_parameter('output_topic', '/calib/pairs_posecov') # Tópico al que se suscribe tu backend
        
        # Parámetros del cubo (movidos desde PairPosePublisher)
        self.declare_parameter('cube_config', '/home/ubuntu/ibima-PostureSense/code/src/multicam_cube_calib/config/cube_config_T.yaml') 
        self.declare_parameter('marker_length', 0.23)
        self.declare_parameter('dictionary', 'DICT_5X5_100')

        # Parámetros de sincronización (¡NUEVOS!)
        self.declare_parameter('sync_slop_sec', 0.02) # 20ms de ventana. AJUSTAR A TU SISTEMA.

        # Parámetros de covarianza (¡NUEVOS!)
        self.declare_parameter('base_variance_m', 0.001) # Varianza base para X/Y
        self.declare_parameter('base_variance_rad', 0.002) # Varianza base para Roll/Pitch/Yaw
        self.declare_parameter('depth_variance_factor', 5.0) # Cuánto más incierta es la profundidad (Z)

        #debug pose covariance visualization
        self.declare_parameter('debug_pose_covariance', True)
        self.declare_parameter('debug_output_topic', '/calib/debug/pose_covariance')

        self.declare_parameter('imu_topic_tpl', '/{}/accel/sample')  # ajusta a tus tópicos reales
        self.declare_parameter('use_camera_accel', True)
        self.declare_parameter('accel_lpf_alpha', 0.15)  # 0..1 (más alto = responde más rápido)

        # ====== Leer parámetros ======
        self.cameras = self.get_parameter('cameras').get_parameter_value().string_array_value
        self.image_topic_tpl = self.get_parameter('image_topic_tpl').get_parameter_value().string_value
        self.caminfo_topic_tpl = self.get_parameter('caminfo_topic_tpl').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.cube_config_path = self.get_parameter('cube_config').value
        self.marker_length = self.get_parameter('marker_length').get_parameter_value().double_value
        self.dict_name = self.get_parameter('dictionary').get_parameter_value().string_value
        self.sync_slop_sec = self.get_parameter('sync_slop_sec').get_parameter_value().double_value

        self.base_var_m = self.get_parameter('base_variance_m').get_parameter_value().double_value
        self.base_var_rad = self.get_parameter('base_variance_rad').get_parameter_value().double_value
        self.depth_var_factor = self.get_parameter('depth_variance_factor').get_parameter_value().double_value

        self.debug_pose_covariance = self.get_parameter('debug_pose_covariance').get_parameter_value().bool_value
        self.debug_output_topic = self.get_parameter('debug_output_topic').get_parameter_value().string_value

        self.imu_topic_tpl = self.get_parameter('imu_topic_tpl').value
        self.use_camera_accel = self.get_parameter('use_camera_accel').value
        self.accel_alpha = self.get_parameter('accel_lpf_alpha').value

        self.get_logger().info(f"Cámaras configuradas: {self.cameras}")


        # ====== QoS / Utilidades ======
        self.bridge = CvBridge()
        self.qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.qos_images = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Cargar diccionario ArUco
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            self.get_logger().error(f"Diccionario ArUco inválido: {self.dict_name}. Usando DICT_5X5_100.")
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Cargar configuración del cubo (¡IMPORTANTE!)
        self.marker_to_cube_tf = self.load_cube_config()
        if not self.marker_to_cube_tf:
            self.get_logger().fatal("No se pudo cargar la configuración del cubo. Abortando.")
            return

        # Puntos 3D del marcador (en su propio frame, Z=0)
        L = self.marker_length / 2.0
        self.objp_marker_local = np.array([[-L, -L, 0], [ L, -L, 0], [ L,  L, 0], [-L,  L, 0]], dtype=np.float32)

        # ====== Estado ======
        self.infos = {}      # cam -> (K, D, (w,h))
        self.info_subs = {}  # cam -> Subscription
        self.imu_subs = {}
        self.sync_started = False
        self.synchronizer = None

        # ====== Publisher ======
        self.pub = self.create_publisher(CameraPairPose, self.output_topic, self.qos_reliable)
        self.get_logger().info(f"Publicando pares en {self.output_topic}")

        if self.debug_pose_covariance:
            self.debug_pub = self.create_publisher(PoseWithCovarianceStamped, self.debug_output_topic, self.qos_reliable)
            self.get_logger().info(f"Publicando debug de pose con covarianza en {self.debug_output_topic}")

        # ====== Iniciar suscripciones de CameraInfo ======
        self.get_logger().info("Esperando CameraInfo de todas las cámaras...")
        for cam in self.cameras:
            info_topic = self.caminfo_topic_tpl.format(cam)
            sub = self.create_subscription(
                CameraInfo, 
                info_topic, 
                lambda msg, c=cam: self.on_info(c, msg), 
                self.qos_images
            )
            self.info_subs[cam] = sub
        self.g_cam = {cam: None for cam in self.cameras}

        if self.use_camera_accel:
            for cam in self.cameras:
                topic = self.imu_topic_tpl.format(cam)
                self.get_logger().info(f"Suscrito a acelerómetro: {topic}")
                sub = self.create_subscription(Imu, topic, lambda msg, c=cam: self.on_accel(c, msg), self.qos_reliable)
                self.imu_subs[cam] = sub

    def load_cube_config(self) -> dict:
        """Carga las poses T_marker_to_cube desde el YAML."""
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
        """Callback para CameraInfo. Almacena K y D."""
        if cam in self.infos:
            return # Ya la tenemos

        try:
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            D = np.array(msg.d, dtype=float)
        except Exception:
            self.get_logger().warning(f"CameraInfo inválido para {cam}; ignorando")
            return

        if msg.width == 0 or np.allclose(K, 0):
            self.get_logger().info(f"CameraInfo incompleto para {cam}; esperando...")
            return

        self.get_logger().info(f"CameraInfo recibido para {cam}.")
        self.infos[cam] = (K, D, (msg.width, msg.height))

        # Cancelar suscripción
        sub = self.info_subs.pop(cam, None)
        if sub:
            self.destroy_subscription(sub)

        # Comprobar si tenemos todas
        self.check_and_start_synchronizer()
    
    def on_accel(self, cam: str, msg: Imu):
        # Muchos drivers ponen -1 en orientation_covariance cuando no es válida ⇒ ignoramos orientation.
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        g = np.array([ax, ay, az], dtype=float)
        n = np.linalg.norm(g)
        if not np.isfinite(g).all() or n < 1e-3:
            return
        g /= n
        # LPF
        if self.g_cam[cam] is None:
            self.g_cam[cam] = g
        else:
            a = float(self.accel_alpha)
            self.g_cam[cam] = (1.0 - a) * self.g_cam[cam] + a * g
            self.g_cam[cam] /= np.linalg.norm(self.g_cam[cam])
        sub = self.imu_subs.pop(cam, None)
        if sub:
            self.destroy_subscription(sub)

        self.check_and_start_synchronizer()
        self.get_logger().info(f"Acelerómetro recibido para {cam}.")

    def check_and_start_synchronizer(self):
        """Si tenemos toda la info, inicia el sincronizador de imágenes. exige que todas las g_cam[cam] sean no-None antes de arrancar. Ejemplo abajo."""
        if self.sync_started or len(self.infos) != len(self.cameras) or not all(self.g_cam[cam] is not None for cam in self.cameras):
            return # Aún no o ya iniciado
        
        self.sync_started = True
        self.get_logger().info("¡Todas las CameraInfos recibidas! Iniciando sincronizador de imágenes.")

        image_subs = []
        for cam in self.cameras:
            img_topic = self.image_topic_tpl.format(cam)
            self.get_logger().info(f"Suscrito a imagen (sync): {img_topic}")
            image_subs.append(message_filters.Subscriber(self, Image, img_topic, qos_profile=self.qos_images))

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            image_subs,
            queue_size=10,
            slop=self.sync_slop_sec
        )
        self.synchronizer.registerCallback(self.synchronized_callback)

    def synchronized_callback(self, *image_msgs: Image):
        """
        ¡Callback principal! Se ejecuta cuando tenemos un conjunto
        de imágenes simultáneas.
        """
        self.get_logger().debug(f"Set de {len(image_msgs)} imágenes sincronizadas recibido.")
        
        poses_with_cov = {} # cam -> (T_cam_cube, Cov_cam_cube)

        # 1. Calcular T_cam_cube y Covarianza para cada cámara
        for cam_name, img_msg in zip(self.cameras, image_msgs):
            K, D, _ = self.infos[cam_name]

            try:
                cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
            except Exception:
                cv_img = self.bridge.imgmsg_to_cv2(img_msg)
                if cv_img.ndim == 3:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            self.get_logger().debug(f"======== Procesando cámara: {cam_name} ========")
            T_cam_cube, Cov_cam_cube = self.estimate_cube_pose_with_covariance(cv_img, K, D)
            
            if T_cam_cube is not None:
                self.get_logger().debug(f"Pose estimada exitosamente para {cam_name}")
                self.get_logger().debug(f"T_cam_cube:\n{T_cam_cube}")
                self.get_logger().debug(f"Diagonal de covarianza: {np.diag(Cov_cam_cube)}")
                poses_with_cov[cam_name] = (T_cam_cube, Cov_cam_cube)
            else:
                self.get_logger().debug(f"No se pudo estimar pose para {cam_name}")

        # 2. Si no hay suficientes detecciones, no hacer nada
        if len(poses_with_cov) < 2:
            self.get_logger().debug(f"Insuficientes detecciones: {len(poses_with_cov)} cámaras (mínimo 2)")
            return

        self.get_logger().debug(f"Generando pares con {len(poses_with_cov)} cámaras: {list(poses_with_cov.keys())}")

        # 3. Generar y publicar todos los pares posibles
        cams = list(poses_with_cov.keys())
        num_pairs = 0
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                ci, cj = cams[i], cams[j]
                
                Ti, Covi = poses_with_cov[ci] # T_ci_cube, Cov_i
                Tj, Covj = poses_with_cov[cj] # T_cj_cube, Cov_j

                self.get_logger().debug(f"======== Procesando par {ci} -> {cj} ========")
                self.get_logger().info(f"Calculando transformación {ci} -> {cj}")
                
                # Calcular par con fusión de acelerómetro
                Tij, Cov_ij = self.compute_pair_pose_fused_with_accel(ci, cj, Ti, Covi, Tj, Covj)
                
                self.get_logger().debug(f"Transformada par {ci}->{cj}:\n{Tij}")
                self.get_logger().debug(f"Diagonal covarianza par: {np.diag(Cov_ij)}")

                # 4. Publicar el mensaje
                self.publish_pair_pose(ci, cj, Tij, Cov_ij, img_msg.header)
                num_pairs += 1

        self.get_logger().info(f"Publicados {num_pairs} pares de cámaras")

    def compute_pair_pose_fused_with_accel(self, ci: str, cj: str,
                                        Ti: np.ndarray, Covi: np.ndarray,
                                        Tj: np.ndarray, Covj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula el transform cam_i -> cam_j fusionando:
        - roll/pitch desde acelerómetros (alinea verticales de cada cámara),
        - yaw desde visión (R_ij_pnp = R_i_c * R_j_c^T),
        y traslación t_ij = t_i - R_ij * t_j.

        Entradas:
        - ci, cj: nombres de cámaras
        - Ti, Tj: T_cam->cube de PnP (4x4)
        - Covi, Covj: covarianzas 6x6 (ROS order [x,y,z, roll,pitch,yaw]) de esas poses

        Salidas:
        - Tij (4x4): cam_i -> cam_j
        - Cov_ij (6x6): covarianza (ROS order)
        """
        self.get_logger().debug(f"Iniciando fusión con acelerómetros para par {ci} -> {cj}")
        
        # ---- Utilidades internas ----
        def _normalize(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-12 else v

        def _rot_align(a, b):
            """Rotación 3x3 que lleva a->b (ambos normalizados)."""
            a = _normalize(a); b = _normalize(b)
            v = np.cross(a, b)
            c = float(np.dot(a, b))
            if c < -0.999999:
                axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                v = _normalize(np.cross(a, axis))
                K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
                return -np.eye(3) + 2*np.outer(v, v)
            K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            return np.eye(3) + K + K @ K * (1.0 / (1.0 + c + 1e-12))

        def _Rz_about(u, angle):
            """Rotación alrededor del eje unitario u (Rodrigues)."""
            u = _normalize(u)
            K = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
            return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)

        # ---- Comprobaciones de sensores ----
        gi = self.g_cam.get(ci, None) if hasattr(self, 'g_cam') else None
        gj = self.g_cam.get(cj, None) if hasattr(self, 'g_cam') else None
        have_accel_both = (gi is not None) and (gj is not None)

        self.get_logger().debug(f"Acelerómetros disponibles: {ci}={'Sí' if gi is not None else 'No'}, {cj}={'Sí' if gj is not None else 'No'}")
        if have_accel_both:
            self.get_logger().debug(f"Vector gravedad {ci}: {gi}")
            self.get_logger().debug(f"Vector gravedad {cj}: {gj}")

        # ---- Rotación relativa de visión (PnP) ----
        Ri = Ti[:3, :3]
        Rj = Tj[:3, :3]
        R_ij_pnp = Ri @ Rj.T  # cam_i -> cam_j desde visión
        self.get_logger().debug(f"R_ij_pnp (solo visión):\n{R_ij_pnp}")

        # ---- Si tenemos acelerómetros en ambas cámaras, fusionamos roll/pitch ----
        if have_accel_both:
            self.get_logger().debug("Aplicando fusión con acelerómetros...")
            # En frame de cada cámara, la "vertical" (up) la tomamos como -g (z_world)
            zi = _normalize(-gi)   # up en cam_i
            zj = _normalize(-gj)   # up en cam_j
            self.get_logger().debug(f"Vertical normalizada {ci}: {zi}")
            self.get_logger().debug(f"Vertical normalizada {cj}: {zj}")

            # 1) Alinear verticales: lleva z_j a z_i (quita roll/pitch relativo)
            R_vert = _rot_align(zj, zi)  # rota en cam_j para que z_j coincida con z_i
            self.get_logger().debug(f"R_vert (alineación vertical):\n{R_vert}")
            
            # 2) Residuo rotacional que queda (idealmente yaw alrededor de z_i)
            R_res = R_ij_pnp @ R_vert.T  # aún en frame de cam_i
            self.get_logger().debug(f"R_res (residuo rotacional):\n{R_res}")

            # 3) Extraer yaw alrededor de z_i:
            #    construimos una base ortonormal {x_i, y_i, z_i}, donde x_i es el "forward" de cam_i
            #    proyectado al plano horizontal (⊥ z_i) para fijar referencia de yaw estable.
            f_i = np.array([0.0, 0.0, 1.0])  # forward en frame cámara i
            x_i = f_i - np.dot(f_i, zi)*zi
            if np.linalg.norm(x_i) < 1e-6:
                # cam mirando exactamente up/down → coge eje auxiliar
                aux = np.array([1.0, 0.0, 0.0]) if abs(zi[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                x_i = aux - np.dot(aux, zi)*zi
            x_i = _normalize(x_i)
            y_i = np.cross(zi, x_i); y_i = _normalize(y_i)
            self.get_logger().debug(f"Base ortonormal - x_i: {x_i}, y_i: {y_i}, z_i: {zi}")

            # Proyecta cómo gira x_i bajo R_res para medir el ángulo en el plano {x_i,y_i}
            x_i_rot = R_res @ x_i
            yaw_num = np.dot(x_i_rot, y_i)
            yaw_den = np.dot(x_i_rot, x_i)
            yaw_i = float(np.arctan2(yaw_num, yaw_den))
            self.get_logger().debug(f"Yaw extraído: {np.degrees(yaw_i):.2f} grados")

            R_yaw = _Rz_about(zi, yaw_i)     # rotación pura alrededor de z_i
            R_ij_fused = R_yaw @ R_vert      # primero quito tilt, luego aplico yaw que viene de visión
            self.get_logger().debug(f"R_ij_fused (con acelerómetros):\n{R_ij_fused}")

        else:
            # Sin acelerómetros en ambas → usa rotación pura de visión
            self.get_logger().debug("Sin acelerómetros disponibles, usando rotación pura de visión")
            R_ij_fused = R_ij_pnp

        # ---- Traslación coherente con la rotación elegida ----
        t_i = Ti[:3, 3]
        t_j = Tj[:3, 3]
        t_ij = t_i - R_ij_fused @ t_j
        self.get_logger().debug(f"Traslación t_i: {t_i}")
        self.get_logger().debug(f"Traslación t_j: {t_j}")
        self.get_logger().debug(f"Traslación t_ij: {t_ij}")

        Tij = np.eye(4)
        Tij[:3, :3] = R_ij_fused
        Tij[:3, 3]  = t_ij

        # ---- Covarianza: parte translacional de visión, parte rotacional fusionada ----
        self.get_logger().debug("Propagando covarianza...")
        # Reutilizamos tu pipeline para propagar, y luego sustituimos la parte rotacional
        Covi_tw = cov_perm(Covi, ['x','y','z','roll','pitch','yaw'], ['roll','pitch','yaw','x','y','z'])
        Covj_tw = cov_perm(Covj, ['x','y','z','roll','pitch','yaw'], ['roll','pitch','yaw','x','y','z'])
        # Nota: usa Tj (como en tu función) para la adjunta; está bien porque traslación se basa en esas Ti/Tj
        Cov_ij_tw_vis = cov_relative_right_invariant(Covi_tw, Covj_tw, Tj)
        Cov_ij = cov_perm(Cov_ij_tw_vis, ['roll','pitch','yaw','x','y','z'], ['x','y','z','roll','pitch','yaw'])
        self.get_logger().debug(f"Covarianza propagada (visión): diag = {np.diag(Cov_ij)}")

        # Sustituir bloque rotacional con confianza alta en roll/pitch si hubo IMU, yaw más laxa
        if have_accel_both and getattr(self, 'use_camera_accel', True):
            rp = max(1e-12, self.base_var_rad * 0.05)
            yw = self.base_var_rad * 5.0
            self.get_logger().debug(f"Ajustando covarianza rotacional: roll/pitch={rp:.6f}, yaw={yw:.6f}")
            Cov_ij[3,3] = rp   # roll
            Cov_ij[4,4] = rp   # pitch
            Cov_ij[5,5] = yw   # yaw
            self.get_logger().debug(f"Covarianza ajustada (con IMU): diag = {np.diag(Cov_ij)}")

        return Tij, Cov_ij

    def estimate_cube_pose_with_covariance(self, cv_img, K, D) -> tuple[np.ndarray, np.ndarray]:
        """
        Resuelve PnP usando TODOS los marcadores vistos y estima una covarianza.
        """
        self.get_logger().debug("Iniciando detección de marcadores ArUco...")
        corners, ids, _ = self.detector.detectMarkers(cv_img)
        
        if ids is None or len(ids) == 0:
            self.get_logger().debug("No se detectaron marcadores ArUco en la imagen")
            return None, None

        self.get_logger().debug(f"Detectados {len(ids)} marcadores: {ids.flatten().tolist()}")

        all_object_points = []
        all_image_points = []
        valid_marker_ids = []

        for marker_id, marker_corners in zip(ids.flatten(), corners):
            T_marker_to_cube = self.marker_to_cube_tf.get(marker_id)
            if T_marker_to_cube is None:
                self.get_logger().debug(f"Marcador {marker_id} no está en la configuración del cubo, ignorando")
                continue # No es un marcador de nuestro cubo

            # Puntos 3D de las esquinas del marcador, en el frame del CUBO
            objp_marker_homog = np.hstack([self.objp_marker_local, np.ones((4, 1))])
            objp_cube_homog = (T_marker_to_cube @ objp_marker_homog.T).T
            
            all_object_points.append(objp_cube_homog[:, :3])
            all_image_points.append(marker_corners.reshape(-1, 2))
            valid_marker_ids.append(marker_id)
            break
            if len(valid_marker_ids) >= 1:
                break  # Usamos solo los dos primeros marcadores válidos para evitar sobrecarga
            
        
        #TODO: Modificar para usar SOLO los marcadores que tengan buena visibilidad segun angulo y distancia
        # Necesitamos al menos 4 puntos (un marcador)
        if len(all_object_points) == 0:
            self.get_logger().debug("No se encontraron marcadores válidos del cubo")
            return None, None
        
        #Debug
        if len(valid_marker_ids) >= 2:
            self.get_logger().debug(f"Marcadores válidos para PnP: {valid_marker_ids}======================================================================\n\n")
            



        self.get_logger().debug(f"Usando {len(valid_marker_ids)} marcadores válidos: {valid_marker_ids}")
            
        final_obj_pts = np.vstack(all_object_points).astype(np.float32)
        final_img_pts = np.vstack(all_image_points).astype(np.float32)
        self.get_logger().debug(f"Total de puntos para PnP: {len(final_obj_pts)} puntos 3D")

        # 1. Resolver PnP robusto UNA SOLA VEZ para T_cam_to_cube
        self.get_logger().debug("Resolviendo PnP...")
        try:
            ok, rvec, tvec = cv2.solvePnP(
                final_obj_pts, final_img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                self.get_logger().debug("solvePnP retornó ok=False")
                return None, None
        except Exception as e:
            self.get_logger().warning(f"solvePnP falló: {e}")
            return None, None

        self.get_logger().debug(f"PnP exitoso - tvec: {tvec.flatten()}, rvec: {rvec.flatten()}")

        # Convertir a matriz 4x4
        R_cam_cube, _ = cv2.Rodrigues(rvec)
        T_cam_cube = np.eye(4)
        T_cam_cube[:3, :3] = R_cam_cube
        T_cam_cube[:3, 3] = tvec.flatten()

        # 2. Estimar Covarianza (Heurística Mejorada)
        self.get_logger().debug("Calculando error de reproyección y covarianza...")
        proj_pts, _ = cv2.projectPoints(final_obj_pts, rvec, tvec, K, D)
        reproj_errors = np.linalg.norm(proj_pts.reshape(-1, 2) - final_img_pts.reshape(-1, 2), axis=1)
        reproj_err_px = np.mean(reproj_errors)
        reproj_err_max = np.max(reproj_errors)
        num_points = len(final_obj_pts)

        self.get_logger().debug(f"Error de reproyección - medio: {reproj_err_px:.3f} px, máximo: {reproj_err_max:.3f} px")

        # La "calidad" es inversamente proporcional al error al cuadrado
        # y proporcional al número de puntos.
        quality = float(num_points) / max(1e-3, reproj_err_px**2)
        self.get_logger().debug(f"Calidad calculada: {quality:.6f} (num_points: {num_points})")

        # La varianza es la inversa de la calidad (escalada por factores base)
        var_scale = 1.0 / max(1e-6, quality)
        
        var_xy = self.base_var_m * var_scale
        var_z  = self.base_var_m * var_scale * self.depth_var_factor # Más incertidumbre en Z
        var_r  = self.base_var_rad * var_scale

        self.get_logger().debug(f"Varianzas finales - var_xy: {var_xy:.6f}, var_z: {var_z:.6f}, var_r: {var_r:.6f}")
        self.get_logger().debug(f"Escala de varianza: {var_scale:.6f}")

        cov_matrix = np.diag([var_xy, var_xy, var_z, var_r, var_r, var_r])
        
        return T_cam_cube, cov_matrix

    def publish_pair_pose(self, ci: str, cj: str, Tij: np.ndarray, Cov_ij: np.ndarray, header):
        """Publica el mensaje final que espera el backend."""
        
        out = CameraPairPose()
        # Usamos el timestamp de la imagen que disparó el callback sincronizado
        out.header.stamp = header.stamp
        out.header.frame_id = ci # Frame 'from'
        
        #Puede que esto sea al reves
        out.camera_from_id = ci
        out.camera_to_id = cj

        pwc = PoseWithCovariance()
        pwc.pose = mat_to_pose(Tij)
        pwc.covariance = Cov_ij.flatten().tolist()
        
        out.measured_transform = pwc
        self.pub.publish(out)

        if self.debug_pose_covariance:
            debug_pose = PoseWithCovarianceStamped()
            debug_pose.header = out.header
            debug_pose.pose = pwc
            self.debug_pub.publish(debug_pose)
 
        self.get_logger().debug(f"Publicado par: {ci} -> {cj}")


def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedPairPublisher()
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