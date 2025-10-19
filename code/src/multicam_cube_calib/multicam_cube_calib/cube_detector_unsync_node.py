#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform

# Mensajes personalizados
from multicam_cube_calib_interfaces.msg import CameraMarker, CamerasMarkersList

# OpenCV / NumPy
from cv_bridge import CvBridge
import numpy as np
import cv2
import copy

# Utilidad SE3->Transform
from multicam_cube_calib.se3 import mat_to_tf


class MultiCameraMarkersNoSync(Node):
    """
    Procesa N cámaras de forma asíncrona (sin sincronización de tópicos).

    - Modo per-image: cuando llega una imagen de una cámara, publica un paquete con esa cámara
      + otras cámaras 'recientes' (edad <= max_age_sec) del buffer. Solo publica si hay
      al menos min_cameras_per_msg cámaras en el paquete.

    - Modo timer: a una tasa fija agrega los últimos resultados 'recientes' y publica
      solo si hay al menos min_cameras_per_msg cámaras (válidas o vacías si include_empty_cameras=True).
    """

    def __init__(self):
        super().__init__('cube_markers_no_sync')

        # ====== Parámetros ======
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01', 'cam01/camera_02', 'cam02/camera_03'])
        self.declare_parameter('image_topic_tpl', '/{}/color/image_raw/decompressed')
        self.declare_parameter('caminfo_topic_tpl', '/{}/color/camera_info')
        self.declare_parameter('output_topic', '/camera_markers_async')

        # Detección ArUco
        self.declare_parameter('marker_length', 0.23)
        self.declare_parameter('dictionary', 'DICT_5X5_100')

        # Publicación (sin sincronía)
        self.declare_parameter('publish_immediately_on_new_image', False)  # modo per-image
        self.declare_parameter('publish_rate_hz', 10.0)                   # usado si publish_immediately_on_new_image=False
        self.declare_parameter('max_age_sec', 0.5)                         # antigüedad máxima para agregación
        self.declare_parameter('include_empty_cameras', False)             # incluir cámaras sin dato en agregación
        self.declare_parameter('min_cameras_per_msg', 2)                   # *** NUEVO: mínimo de cámaras por mensaje ***

        # ====== Leer parámetros ======
        self.cameras = list(self.get_parameter('cameras').get_parameter_value().string_array_value)
        self.image_topic_tpl = self.get_parameter('image_topic_tpl').get_parameter_value().string_value
        self.caminfo_topic_tpl = self.get_parameter('caminfo_topic_tpl').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.marker_length = float(self.get_parameter('marker_length').get_parameter_value().double_value)
        self.dict_name = self.get_parameter('dictionary').get_parameter_value().string_value

        self.publish_immediately = self.get_parameter('publish_immediately_on_new_image').get_parameter_value().bool_value
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').get_parameter_value().double_value)
        self.max_age_sec = float(self.get_parameter('max_age_sec').get_parameter_value().double_value)
        self.include_empty_cameras = self.get_parameter('include_empty_cameras').get_parameter_value().bool_value
        self.min_cams_per_msg = int(self.get_parameter('min_cameras_per_msg').get_parameter_value().integer_value or 2)

        # ====== QoS / utilidades ======
        self.bridge = CvBridge()
        self.qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))
        except AttributeError:
            self.get_logger().error(f"Diccionario ArUco inválido: {self.dict_name}. Usando DICT_5X5_100 por defecto.")
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        # OpenCV >=4.7
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception:
            # Fallback para versiones anteriores
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = None

        # ====== Estado ======
        self.infos = {}          # cam -> (K, D, (w,h))
        self.info_subs = {}      # para cancelar tras recibir info válida
        self.last_results = {}   # cam -> (CameraMarker, Time stamp)

        # ====== Subs CameraInfo ======
        for cam in self.cameras:
            info_topic = self.caminfo_topic_tpl.format(cam)
            sub = self.create_subscription(CameraInfo, info_topic, lambda msg, c=cam: self.on_info(c, msg), self.qos)
            self.info_subs[cam] = sub
            self.get_logger().info(f"Suscrito a CameraInfo: {info_topic}")

        # ====== Subs Image (sin sincronizar) ======
        for cam in self.cameras:
            img_topic = self.image_topic_tpl.format(cam)
            self.create_subscription(Image, img_topic, lambda msg, c=cam: self.on_image(c, msg), self.qos)
            self.get_logger().info(f"Suscrito a imagen (async): {img_topic}")

        # ====== Publisher ======
        self.pub = self.create_publisher(CamerasMarkersList, self.output_topic, 10)
        self.get_logger().info(f"Publicando CamerasMarkersList en {self.output_topic}")

        # ====== Timer de agregación (opcional) ======
        self.timer = None
        if not self.publish_immediately and self.publish_rate_hz > 0.0:
            period = 1.0 / self.publish_rate_hz
            self.timer = self.create_timer(period, self.on_timer)
            self.get_logger().info(
                f"Agregación por temporizador @ {self.publish_rate_hz:.2f} Hz "
                f"(max_age={self.max_age_sec}s, min_cams={self.min_cams_per_msg})"
            )

    # ================== Callbacks ==================

    def on_info(self, cam: str, msg: CameraInfo):
        # Validación básica
        try:
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            D = np.array(msg.d, dtype=float)
        except Exception:
            self.get_logger().warning(f"CameraInfo inválido para {cam}; ignorando")
            return

        if msg.width == 0 or msg.height == 0 or np.allclose(K, 0):
            self.get_logger().info(f"CameraInfo incompleto para {cam}; esperando...")
            return

        self.infos[cam] = (K, D, (msg.width, msg.height))

        # Cancelar sub para esa cámara
        sub = self.info_subs.get(cam)
        if sub is not None:
            try:
                self.destroy_subscription(sub)
                del self.info_subs[cam]
                self.get_logger().info(f"Cancelada suscripción CameraInfo de {cam} tras recibir info válida")
            except Exception as e:
                self.get_logger().warning(f"No se pudo destruir suscripción CameraInfo de {cam}: {e}")

    def on_image(self, cam: str, msg: Image):
        # Si no tenemos intrínsecos de esa cámara aún, no procesar
        if cam not in self.infos:
            return

        K, D, _ = self.infos[cam]

        # Convertir imagen a gris
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception:
            cv_img = self.bridge.imgmsg_to_cv2(msg)
            if cv_img.ndim == 3:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Detección ArUco
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(cv_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(cv_img, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            self.get_logger().debug(f"Cámara {cam} detectó los marcadores: {ids.flatten().tolist()}")
        # Construir CameraMarker para esta cámara
        cam_marker_msg = self._build_camera_marker(cam, msg.header, K, D, corners, ids)

        # Guardar último resultado para esta cámara
        now_stamp = self.get_clock().now()
        self.last_results[cam] = (cam_marker_msg, now_stamp)

        if self.publish_immediately:
            # Armar paquete con esta cámara + otras 'recientes' (edad <= max_age_sec)
            cameras_list = [cam_marker_msg]
            now = self.get_clock().now()
            for other in self.cameras:
                if other == cam:
                    continue
                entry = self.last_results.get(other)
                if entry is None:
                    continue
                cm, t = entry
                age = (now - t).nanoseconds / 1e9
                if age <= self.max_age_sec:
                    cameras_list.append(copy.deepcopy(cm))

            if len(cameras_list) >= self.min_cams_per_msg:
                out = CamerasMarkersList()
                out.header = Header()
                # Usamos el stamp de la imagen que disparó el paquete
                out.header.stamp = msg.header.stamp
                out.header.frame_id = ''
                out.cameras = cameras_list
                self.pub.publish(out)
            else:
                # No hay suficientes cámaras 'recientes' para formar pareja
                self.get_logger().info(
                    f"Skip publish: solo {len(cameras_list)} cámara(s) disponible(s) <= {self.max_age_sec}s."
                )

    def on_timer(self):
        #Si no hay info de cámaras, no publicar
        if len(self.last_results) == 0:
            self.get_logger().debug("No hay datos de cámaras para agregar en timer.")
            return
        now = self.get_clock().now()
        cameras_list = []
        valid_count = 0

        for cam in self.cameras:
            entry = self.last_results.get(cam, None)
            if entry is not None:
                cm, t = entry
                age = (now - t).nanoseconds / 1e9
                if age <= self.max_age_sec:
                    cameras_list.append(copy.deepcopy(cm))
                    valid_count += 1
                elif self.include_empty_cameras:
                    cameras_list.append(self._empty_camera_marker(cam, now))
            else:
                if self.include_empty_cameras:
                    cameras_list.append(self._empty_camera_marker(cam, now))

        # Condición de publicación: al menos min_cams_per_msg cámaras
        # (válidas; o contando vacías si include_empty_cameras=True)
        can_publish = (
            valid_count >= self.min_cams_per_msg
            or (self.include_empty_cameras and len(cameras_list) >= self.min_cams_per_msg)
        )

        if can_publish and len(cameras_list) > 0:
            out = CamerasMarkersList()
            out.header = Header()
            out.header.stamp = now.to_msg()
            out.header.frame_id = ''
            out.cameras = cameras_list
            self.pub.publish(out)
        else:
            self.get_logger().debug(
                f"Skip timer publish: válidas={valid_count}, total={len(cameras_list)}, "
                f"min_cams={self.min_cams_per_msg}"
            )

    # ================== Utilidades ==================

    def _empty_camera_marker(self, cam: str, now_time: Time):
        cm = CameraMarker()
        cm.header = Header()
        cm.header.stamp = now_time.to_msg()
        cm.header.frame_id = f"{cam}_optical"
        cm.camera = cam
        cm.marker_ids = []
        cm.t_cam_marker = []
        cm.confidence = []
        return cm

    def _build_camera_marker(self, cam: str, header: Header, K, D, corners, ids):
        cm = CameraMarker()
        cm.header = Header()
        cm.header.stamp = header.stamp
        cm.header.frame_id = f"{cam}_optical"
        cm.camera = cam

        if ids is None or len(ids) == 0:
            cm.marker_ids = []
            cm.t_cam_marker = []
            cm.confidence = []
            return cm

        ids = ids.flatten().tolist()

        # Cuadrado ArUco en el plano Z=0 (centro en el origen)
        L = self.marker_length / 2.0
        objp = np.array([[-L, -L, 0], [ L, -L, 0], [ L,  L, 0], [-L,  L, 0]], dtype=np.float32)

        marker_ids = []
        transforms = []
        confidences = []

        for i, corners_i in zip(ids, corners):
            imgp = corners_i.reshape(-1, 2).astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue

            R_cf, _ = cv2.Rodrigues(rvec)
            T_c_m = np.eye(4)
            T_c_m[:3, :3] = R_cf
            T_c_m[:3, 3] = tvec.flatten()

            # Confianza simple basada en error de reproyección y distancia
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
            proj = proj.reshape(-1, 2)
            reproj_err = float(np.linalg.norm(proj - imgp, axis=1).mean())
            dist = float(np.linalg.norm(tvec))
            conf = 1.0 / (1.0 + reproj_err)
            if 0.3 <= dist <= 1.5:
                conf *= 1.2
            conf = float(np.clip(conf, 0.01, 1.0))

            marker_ids.append(int(i))
            transforms.append(mat_to_tf(T_c_m))  # geometry_msgs/Transform
            confidences.append(conf)

        cm.marker_ids = marker_ids
        cm.t_cam_marker = transforms
        cm.confidence = confidences
        return cm


def main():
    rclpy.init()
    node = MultiCameraMarkersNoSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
