# Versión refinada del sistema de detección ArUco robusto para calibración extrínseca
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import time

class Marker:
    def __init__(self, marker_id, max_history=50, alpha=0.1):
        self.id = marker_id
        self.tvec = None
        self.rvec = None
        self.area = None
        self.tf = None  # This will be set when the marker is reliable
        self.state = 'pending'  # pending, reliable, rejected, dead
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.history = deque(maxlen=max_history)
        self.alpha = alpha
        self.score = 0.0

    def add_observation(self, tvec, rvec, area):
        ts = time.time()
        self.history.append({'tvec': np.array(tvec).flatten(), 'rvec': np.array(rvec).flatten(), 'area': area, 'ts': ts})
        self.last_seen = ts
        self._update_score()

    def _update_score(self):
        n = len(self.history)
        score_count = min(1.0, n / 10.0)
        tvecs = np.stack([h['tvec'] for h in self.history])
        areas = np.array([h['area'] for h in self.history])
        var_t = np.mean(np.var(tvecs, axis=0))
        var_a = float(np.var(areas))
        var_score = np.exp(-(var_t + var_a) / 500.0)
        combined = 0.5 * score_count + 0.5 * var_score
        self.score = (1 - self.alpha) * self.score + self.alpha * combined

    def evaluate(self, reliable_thresh=0.85, reject_var_thresh=500.0, dead_timeout=10.0):
        if self.state in ['reliable', 'rejected', 'dead']:
            return self.state
        age = time.time() - self.first_seen
        if self.score >= reliable_thresh:
            self.state = 'reliable'
            tvec_avg = np.stack([h['tvec'] for h in self.history])
            self.tvec = np.mean(tvec_avg, axis=0)
            rvec_avg = np.stack([h['rvec'] for h in self.history])
            self.rvec = np.mean(rvec_avg, axis=0)
            area_avg = np.mean([h['area'] for h in self.history])
            self.area = area_avg
            self.tf = np.eye(4)
            self.tf[:3, :3] = cv2.Rodrigues(self.rvec)[0]
            self.tf[:3, 3] = self.tvec

        elif len(self.history) >= self.history.maxlen and \
                (np.mean(np.var(np.stack([h['tvec'] for h in self.history]), axis=0)) + np.var([h['area'] for h in self.history])) > reject_var_thresh:
            self.state = 'rejected'
        elif age > dead_timeout:
            self.state = 'dead'
        return self.state

class MarkerTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_length):
        self.K = camera_matrix
        self.D = dist_coeffs
        self.L = marker_length
        self.states = {}
        self.reliable_markers = {}
        h = marker_length / 2
        self.objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def process(self, corners, ids, markers : dict):
        if ids is None:
            return False
        for c, id_arr in zip(corners, ids.flatten()):
            id_ = int(id_arr)

            marker = markers.setdefault(id_, Marker(id_))
            if marker.state in ['rejected', 'dead', 'reliable']:
                continue

            succ, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.objp, c, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=4.0, iterationsCount=100, confidence=0.99)
            if not succ:
                continue

            area = cv2.contourArea(c.reshape(4, 2))

            marker.add_observation(tvec, rvec, area)
            [marker.evaluate() for marker in markers.values()]

        return True


class Camera:
    def __init__(self, node: Node, camera_name: str, camera_id: int, image_topic: str, camera_info_topic: str,
                 marker_length: float, aruco_dict_name: str,camera_frame_id: str = None):
        self.node = node
        self.camera_name = camera_name
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.marker_length = marker_length
        self.bridge = CvBridge()
        self.camera_id = camera_id
        self.camera_frame_id = camera_frame_id 
        self.can_camera_connect_two_markers_table = None

        self.camera_matrix = None
        self.dist_coeffs = None
        self.markers = {}

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.image_sub = self.node.create_subscription(Image, image_topic, self.image_callback, 1)
        self.camera_info_sub = self.node.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        self.cv2_image_publisher = self.node.create_publisher(Image, f"{image_topic}/debug", 10)

        self.marker_tracker = None
        self.last_seen_time = time.time()

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)
        self.marker_tracker = MarkerTracker(self.camera_matrix, self.dist_coeffs, self.marker_length)
        self.node.destroy_subscription(self.camera_info_sub)
        self.camera_info_sub = None

    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        current_time = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.GaussianBlur(gray, (5, 5), 0)

        corners, ids, _ = self.detector.detectMarkers(gray_filtered)

        if ids is None or len(ids) == 0:
            if current_time - self.last_seen_time > 10.0:
                self.node.get_logger().fatal(f"{self.camera_name}: No markers detected in the last 10 seconds. Shutting down.")
            return
        self.last_seen_time = current_time

        self.marker_tracker.process(corners, ids, self.markers)


    def are_all_transforms_precise(self, precise_thresh=0.85) -> bool:
        if not self.markers:
            self.node.get_logger().warn(f"{self.camera_name}: No hay marcadores activos.")
            return False
                # Mostrar el diccionario de marcadores de forma visual para debug
        # for id_, marker in self.markers.items():
            # self.node.get_logger().info(f"{self.camera_name}: Marcador {id_} - Estado: {marker.state}")
        unreliable = [id_ for id_, marker  in self.markers.items() if marker.score < precise_thresh and marker.state not in ['reliable', 'rejected', 'dead']]
        if unreliable:
            self.node.get_logger().warn(f"{self.camera_name}: Marcadores no precisos: {unreliable}")
            return False

        self.node.get_logger().info(f"{self.camera_name}: Todos los marcadores son precisos.")
        return True
    
    def _remove_unreliable_markers(self):
        to_remove = [id_ for id_, marker in self.markers.items() if marker.state not in ['reliable']]
        for id_ in to_remove:
            del self.markers[id_]
            self.node.get_logger().info(f"{self.camera_name}: Marcador {id_} eliminado por no ser confiable.")
