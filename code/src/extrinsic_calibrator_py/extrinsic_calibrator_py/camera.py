# Implementación de la clase Camera integrando umbrales adaptativos, filtrado, scoring, rescate y timeout adaptativo
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from .marker import Marker
import tf2_ros
import time


class Camera:
    def __init__(self, node: Node, camera_name: str, camera_id: int, image_topic: str, camera_info_topic: str,
                 marker_length: float, aruco_dict_name: str, camera_frame_id: str, verbose=True):
        self.node = node
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.bridge = CvBridge()
        self.verbose = verbose
        self.camera_frame_id = camera_frame_id

        # Umbrales base y adaptativos
        self.base_distance_threshold = 5.5
        self.base_area_threshold = 300
        self.adaptive_distance_threshold = self.base_distance_threshold
        self.adaptive_area_threshold = self.base_area_threshold
        # Factores de margen (por ejemplo, permitir hasta un 20% más de distancia y un 20% menor de área)
        self.distance_factor = 1.2
        self.area_factor = 0.8

        self.diff_threshold = 0.1  # Para usar en el cálculo del timeout adaptativo

        self.rejected_timeout = 3.0    # Tiempo de rechazo en segundos
        # Tiempo máximo (en segundos) para intentar que un marcador se vuelva reliable
        self.max_attempt_time = 60.0

        self.node.get_logger().info(f"Camera {self.camera_name} created.")

        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_length = marker_length

        # Configuración de detección ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, aruco_dict_name))
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.parameters)

        # Precomputar los puntos objeto del marcador (suponiendo un marcador centrado y cuadrado)
        half_length = self.marker_length / 2.0
        self.obj_points = np.array([[-half_length,  half_length, 0],
                                    [half_length,  half_length, 0],
                                    [half_length, -half_length, 0],
                                    [-half_length, -half_length, 0]], dtype=np.float32)

        # Subscripciones y publicación de imágenes con detecciones
        self.image_sub = self.node.create_subscription(
            Image, image_topic, self.image_callback, 1)
        self.camera_info_sub = self.node.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        self.cv2_image_publisher = self.node.create_publisher(
            Image, f"{image_topic}/detected_markers", 10)

        self.markers = {}         # Marcadores válidos: {marker_id: Marker}
        self.rejected_markers = {}  # Marcadores rechazados: {marker_id: timestamp_rechazo}
        self.dead_markers = {}      # Marcadores que excedieron el tiempo máximo de intento
        self.node.get_logger().info(
            f"Camera {self.camera_name} initialized with ArUco parameters: {aruco_dict_name}, marker length: {marker_length}m.")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.node.get_logger().info(
                f"Camera {self.camera_name} parameters received.")

    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.node.get_logger().warn(
                f"Camera {self.camera_name} parameters not yet received.")
            return

        current_time = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)

        corners, ids, _ = self.detector.detectMarkers(gray_filtered)
        if ids is None:
            return

        for i, id_arr in enumerate(ids):
            marker_id = int(id_arr[0])
            # Omitir marcadores que ya han sido marcados como dead
            if marker_id in self.dead_markers:
                self.node.get_logger().debug(
                    f"Camera {self.camera_name}: Marcador {marker_id} marcado como dead, se omite.")
                continue

            corner = corners[i].reshape((4, 2))
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points, corners[i], self.camera_matrix, self.dist_coeffs)
            if not success:
                continue

            distance = np.linalg.norm(tvec)
            area = cv2.contourArea(corner)

            if distance < 0.1 or area < 10 or distance > 100 or area > 10000:
                self.node.get_logger().warn(
                    f"Camera {self.camera_name}: Marcador {marker_id} descartado por distancia o área fuera de rango.")
                continue

            # Validación adaptativa del marcador
            if not self._is_marker_valid(distance, area):
                self._reject_marker(marker_id, current_time)
                if marker_id in self.markers:
                    self.node.get_logger().info(
                        f"Camera {self.camera_name}: Marcador {marker_id} removido de la lista de válidos.")
                    # self.node.get_logger().info(f"Camera {self.camera_name}: Distancia: {distance:.2f}, Área: {area:.2f}")
                    del self.markers[marker_id]
                continue
            else:
                # Actualizar umbrales adaptativos con la detección válida
                # self._update_adaptive_thresholds(distance, area)
                # "Rescatar" el marcador si estaba en la lista de rechazados y ya pasó el timeout de rechazo
                if marker_id in self.rejected_markers and (current_time - self.rejected_markers[marker_id] > self.rejected_timeout):
                    del self.rejected_markers[marker_id]

                if marker_id not in self.markers:
                    self.markers[marker_id] = Marker(
                        marker_id, self.marker_length, timeout=5.0)
                # Convertir la solución de PnP en una matriz de transformación 4x4
                rot_matrix, _ = cv2.Rodrigues(rvec)
                translation_matrix = np.eye(4)
                translation_matrix[:3, :3] = rot_matrix
                translation_matrix[:3, 3] = tvec.flatten()
                self.markers[marker_id].update(translation_matrix)

                if self.verbose:
                    cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    cv2.drawFrameAxes(
                        cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length / 2)
                    try:
                        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                        self.cv2_image_publisher.publish(ros_image)
                    except Exception as e:
                        self.node.get_logger().error(
                            f"Camera {self.camera_name}: Error publishing image: {e}")
        # self.node.get_logger().info(f"Parametros adaptativos: Distancia: {self.adaptive_distance_threshold:.2f}, Área: {self.adaptive_area_threshold:.2f}")
        self._update_marker_status(current_time)

    def _is_marker_valid(self, distance: float, area: float) -> bool:
        valid_distance = distance <= self.adaptive_distance_threshold * self.distance_factor
        valid_area = area >= self.adaptive_area_threshold * self.area_factor
        return valid_distance and valid_area

    def _reject_marker(self, marker_id: int, current_time: float):
        self.rejected_markers[marker_id] = current_time

    def _update_adaptive_thresholds(self, distance: float, area: float, alpha: float = 0.1):
        # Actualiza los umbrales adaptativos usando un promedio móvil exponencial
        self.adaptive_distance_threshold = (
            1 - alpha) * self.adaptive_distance_threshold + alpha * distance
        self.adaptive_area_threshold = (
            1 - alpha) * self.adaptive_area_threshold + alpha * area

    def _update_marker_status(self, current_time: float):
        for marker_id in list(self.markers):
            marker = self.markers[marker_id]
            # Calcular un timeout efectivo en función de la variabilidad: a mayor inestabilidad, menor timeout
            ratio = min(marker.variability / self.diff_threshold,
                        0.5) if self.diff_threshold > 0 else 0
            effective_timeout = self.max_attempt_time * (1 - ratio)
            if marker.is_timed_out():
                self.node.get_logger().warn(
                    f"Camera {self.camera_name}: Marcador {marker_id} timed out, removiendo de tracking.")
                del self.markers[marker_id]
            elif not marker.reliable:
                if marker.is_precise():
                    marker.reliable = True
                    self.node.get_logger().info(
                        f"Camera {self.camera_name}: Marcador {marker_id} ahora es reliable.")
                elif (current_time - marker.first_seen) > effective_timeout:
                    self.node.get_logger().warn(
                        f"Camera {self.camera_name}: El marcador {marker_id} excedió el tiempo máximo de intento para volverse reliable. Marcándolo como dead."
                    )
                    self.dead_markers[marker_id] = current_time
                    del self.markers[marker_id]

    def are_all_transforms_precise(self) -> bool:
        if not self.markers:
            self.node.get_logger().warn(
                f"Camera {self.camera_name}: No se han detectado marcadores.")
            return False

        unreliable_markers = [
            marker.id for marker in self.markers.values() if not marker.is_precise()]

        if unreliable_markers:
            self.node.get_logger().warn(
                f"Camera {self.camera_name}: Los siguientes marcadores aún no son reliable: {', '.join(map(str, unreliable_markers))}."
            )
            return False
        else:
            self.node.get_logger().info(
                f"Camera {self.camera_name}: ¡Todos los marcadores son reliable!")
            return True
