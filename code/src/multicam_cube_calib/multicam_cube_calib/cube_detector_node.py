#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform

# NUEVOS mensajes agregados
from multicam_cube_calib_interfaces.msg import CameraMarker, CamerasMarkersList

# OpenCV / NumPy
from cv_bridge import CvBridge
import numpy as np
import cv2

# Sincronizador temporal ROS2
import message_filters

# Utilidades (mantenemos la misma helper para convertir SE3->Transform)
from multicam_cube_calib.se3 import mat_to_tf

class MultiCameraMarkersSync(Node):
    """
    Sincroniza N cámaras en el tiempo y publica, para cada 'tick' sincronizado,
    un único mensaje CamerasMarkersList con un CameraMarker por cámara.
    """
    def __init__(self):
        super().__init__('cube_markers_sync')

        # Parámetros
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01', 'cam01/camera_02', 'cam02/camera_03'])
        self.declare_parameter('image_topic_tpl', '/{}/color/image_raw/decompressed')
        self.declare_parameter('caminfo_topic_tpl', '/{}/color/camera_info')
        self.declare_parameter('output_topic', '/camera_markers_sync')

        # Sincronizador
        self.declare_parameter('approximate', True)        # True = ApproximateTimeSynchronizer
        self.declare_parameter('sync_queue_size', 10)      # cola del sincronizador
        self.declare_parameter('sync_slop_sec', 0.2)      # tolerancia (s) si approximate=True
        self.declare_parameter('marker_length', 0.23)        # longitud del marcador (m)
        self.declare_parameter('dictionary', 'DICT_5X5_100')  # diccionario ArUco

        self.cameras = list(self.get_parameter('cameras').get_parameter_value().string_array_value)
        self.image_topic_tpl = self.get_parameter('image_topic_tpl').get_parameter_value().string_value
        self.caminfo_topic_tpl = self.get_parameter('caminfo_topic_tpl').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.approximate = self.get_parameter('approximate').get_parameter_value().bool_value
        self.sync_queue_size = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        self.sync_slop = float(self.get_parameter('sync_slop_sec').get_parameter_value().double_value)
        self.marker_length = float(self.get_parameter('marker_length').get_parameter_value().double_value)
        self.dict_name = self.get_parameter('dictionary').get_parameter_value().string_value


        # QoS
        self.bridge = CvBridge()
        self.qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dict_name))
        except AttributeError:
            self.get_logger().error(f"Diccionario ArUco inválido: {self.dict_name}. Usando DICT_5X5_100 por defecto.")
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)


        # Estados de intrínsecos por cámara
        self.infos = {}  # cam -> (K, D, (w,h))

        # Suscripciones a CameraInfo (no hay que sincronizarlas, solo cache)
        # Guardamos los objetos Subscription para poder destruirlos cuando recibamos
        # una CameraInfo válida y ya no necesitemos más mensajes de esa cámara.
        self.info_subs = {}
        for cam in self.cameras:
            info_topic = self.caminfo_topic_tpl.format(cam)
            sub = self.create_subscription(CameraInfo, info_topic, lambda msg, c=cam: self.on_info(c, msg), self.qos)
            self.info_subs[cam] = sub
            self.get_logger().info(f"Suscrito a CameraInfo: {info_topic}")

        # Subs de imagen + sincronizador
        self.img_subs = []
        for cam in self.cameras:
            img_topic = self.image_topic_tpl.format(cam)
            sub = message_filters.Subscriber(self, Image, img_topic, qos_profile=self.qos)
            self.img_subs.append(sub)
            self.get_logger().info(f"Suscrito a imagen: {img_topic}")

        if self.approximate:
            self.sync = message_filters.ApproximateTimeSynchronizer(
                self.img_subs, queue_size=self.sync_queue_size, slop=self.sync_slop, allow_headerless=False
            )
        else:
            self.sync = message_filters.TimeSynchronizer(self.img_subs, queue_size=self.sync_queue_size)

        self.sync.registerCallback(self.synced_images_cb)

        # Publisher agregado
        self.pub = self.create_publisher(CamerasMarkersList, self.output_topic, 10)
        self.get_logger().info(f"Publicando CamerasMarkersList en {self.output_topic}")

    # ==== Callbacks ===========================================================

    def on_info(self, cam: str, msg: CameraInfo):
        # Validación básica: descartamos CameraInfo vacíos o con K nulo
        try:
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            D = np.array(msg.d, dtype=float)
        except Exception:
            self.get_logger().warning(f"CameraInfo inválido recibido para {cam}; ignorando")
            return

        if msg.width == 0 or msg.height == 0 or np.allclose(K, 0):
            # No es una info útil aún
            self.get_logger().info(f"CameraInfo incompleto para {cam} (w/h/K vacíos); esperando...")
            return

        # Guardamos intrínsecos y eliminamos la suscripción para evitar recibir más msgs
        self.infos[cam] = (K, D, (msg.width, msg.height))

        # Si tenemos la suscripción guardada, la destruimos para no procesar más CameraInfo
        sub = self.info_subs.get(cam)
        if sub is not None:
            try:
                self.destroy_subscription(sub)
                del self.info_subs[cam]
                self.get_logger().info(f"Destroyed CameraInfo subscription for {cam} after receiving valid info")
            except Exception as e:
                # Log y continuar; la ausencia de destrucción no es crítica
                self.get_logger().warning(f"No se pudo destruir suscripción CameraInfo de {cam}: {e}")

    def synced_images_cb(self, *image_msgs: Image):
        self.get_logger().info(f"Recibidos {len(image_msgs)} imágenes sincronizadas")
        # Verificar que tenemos intrínsecos de todas las cámaras
        missing = [c for c in self.cameras if c not in self.infos]
        if missing:
            # Aún no tenemos CameraInfo de todas; ignoramos este tick
            return

        # Emparejar (cam_name, image_msg) respetando el orden de self.cameras
        cam_msgs = list(zip(self.cameras, image_msgs))

        cameras_list = []
        stamps = []

        for cam, msg in cam_msgs:
            stamps.append(msg.header.stamp)
            K, D, _ = self.infos[cam]

            # Imagen en gris
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            except Exception:
                cv_img = self.bridge.imgmsg_to_cv2(msg)  # passthrough
                if cv_img.ndim == 3:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # Detección de marcadores
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(cv_img)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(cv_img, self.aruco_dict, parameters=self.aruco_params)

            #Loggear detecciones
            if ids is not None and len(ids) > 0:
                self.get_logger().info(f"Cámara {cam}: detectados {len(ids)} marcadores: {ids.flatten().tolist()}")
            else:
                self.get_logger().info(f"Cámara {cam}: no se detectaron marcadores")

            cam_marker_msg = CameraMarker()
            cam_marker_msg.header = Header()
            cam_marker_msg.header.stamp = msg.header.stamp
            cam_marker_msg.header.frame_id = f"{cam}_optical"
            cam_marker_msg.camera = cam

            if ids is None or len(ids) == 0:
                # Incluimos la cámara con arrays vacíos (procesada sin detecciones)
                cameras_list.append(cam_marker_msg)
                continue

            ids = ids.flatten().tolist()

            # solvePnP por marcador (mismo pipeline)
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

                # Error de reproyección -> confianza
                proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
                proj = proj.reshape(-1, 2)
                reproj_err = float(np.linalg.norm(proj - imgp, axis=1).mean())
                dist = float(np.linalg.norm(tvec))
                conf = 1.0 / (1.0 + reproj_err)
                if 0.3 <= dist <= 1.5:
                    conf *= 1.2
                conf = float(np.clip(conf, 0.01, 1.0))

                # Rellenar arrays del mensaje
                marker_ids.append(int(i))
                transforms.append(mat_to_tf(T_c_m))  # geometry_msgs/Transform
                confidences.append(conf)

            cam_marker_msg.marker_ids = marker_ids
            cam_marker_msg.t_cam_marker = transforms  # el campo se llama t_cam_marker en tu msg
            cam_marker_msg.confidence = confidences

            cameras_list.append(cam_marker_msg)

        # Mensaje agregado
        out_msg = CamerasMarkersList()
        out_msg.header = Header()
        # Usamos el stamp del primer mensaje sincronizado
        out_msg.header.stamp = stamps[0] if stamps else self.get_clock().now().to_msg()
        out_msg.header.frame_id = ''  # opcional
        out_msg.cameras = cameras_list

        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = MultiCameraMarkersSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
