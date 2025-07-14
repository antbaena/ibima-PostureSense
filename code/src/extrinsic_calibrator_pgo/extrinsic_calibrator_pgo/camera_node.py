# extrinsic_calibrator_pgo/camera_node.py

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from extrinsic_calibrator_interfaces.msg import MarkerObservation
from tf_transformations import quaternion_from_matrix

from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # === Parámetros configurables ===
        self.declare_parameter('camera_id', 'cam_1')
        self.declare_parameter('marker_size', 0.05)  # metros
        self.declare_parameter('aruco_dict', 'DICT_5X5_50')
        self.declare_parameter('depth_filter_max', 2.5)
        self.declare_parameter('quality_threshold', 0.01)
        self.declare_parameter('image_topic', '/image_color')
        self.declare_parameter('depth_topic', '/image_depth')
        self.declare_parameter('camera_info_topic', '/camera_info')

        self.camera_id = self.get_parameter('camera_id').get_parameter_value().string_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        self.aruco_dict_name = self.get_parameter('aruco_dict').get_parameter_value().string_value
        self.depth_filter_max = self.get_parameter('depth_filter_max').get_parameter_value().double_value
        self.quality_threshold = self.get_parameter('quality_threshold').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        # === Comunicación ===
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_callback, 10)
        # self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.pub = self.create_publisher(MarkerObservation, '/marker_observations', 10)

        # === Internos ===
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_image = None
        self.aruco_params = None
        try:
            self.aruco_dict_id = getattr(aruco, self.aruco_dict_name)
            self.aruco_dict = aruco.getPredefinedDictionary(self.aruco_dict_id)
            self.aruco_params = aruco.DetectorParameters()
            self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        except AttributeError:
            self.get_logger().error(f"Aruco dictionary {self.aruco_dict_name} no es válido.")
            raise


        self.get_logger().info(f"[{self.camera_id}] Nodo inicializado suscribiendo a {self.image_topic} y {self.camera_info_topic}")

    def info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.destroy_subscription(self.info_sub)  # Solo necesitamos esto una vez

    def depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return  # Esperar a tener todo

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]

            # Leer profundidad del centro del marcador
            center_px = np.mean(corners[i][0], axis=0).astype(int)
            # z = self.depth_image[center_px[1], center_px[0]] / 1000.0  # mm → m
            z = 1

            if z == 0 or z > self.depth_filter_max:
                continue  # ruido o fuera de rango
            # Confianza simple basada en profundidad
            confidence = compute_confidence(z, rvec, corners[i][0], frame)
            # if confidence < self.quality_threshold:
            #     continue

            # Publicar transform
            marker_corners = corners[i][0]
            tf_msg = self.build_transform_msg(marker_id, tvec, rvec, msg.header, confidence, z, marker_corners, msg.header.frame_id)
            self.get_logger().info(f"[{self.camera_id}] Publicando marcador {marker_id} con confianza {confidence:.2f}")
            self.pub.publish(tf_msg)

    def build_transform_msg(self, marker_id, tvec, rvec, header, confidence, z, corners, frame_id):
        observation = MarkerObservation()

        observation.header.stamp = header.stamp
        observation.header.frame_id = frame_id
        observation.marker_id = int(marker_id)
        observation.camera_id = self.camera_id

        observation.transform.translation.x = float(tvec[0][0])
        observation.transform.translation.y = float(tvec[0][1])
        observation.transform.translation.z = float(tvec[0][2])

        rot_matrix, _ = cv2.Rodrigues(rvec)
        M = np.eye(4)
        M[:3, :3] = rot_matrix
        quat =  quaternion_from_matrix(M)

        observation.transform.rotation.x = quat[0]
        observation.transform.rotation.y = quat[1]
        observation.transform.rotation.z = quat[2]
        observation.transform.rotation.w = quat[3]

        observation.confidence = float(confidence)
        observation.distance = float(z)
        observation.pixel_area = float(cv2.contourArea(corners))

        return observation
    
def compute_confidence(z, rvec, corner_pts, frame):
    w1, w2, w3, w4, w5 = 0.3, 0.2, 0.2, 0.2, 0.1
    return (
        w1 * f_distance(z) +
        w2 * f_angle(rvec) +
        w3 * f_size(corner_pts, frame.shape) +
        w4 * f_sharpness(frame, corner_pts) +
        w5 * f_corners_detected(corner_pts)
    )
def f_distance(z, max_distance=2.5):
    if z == 0 or z > max_distance:
        return 0.0
    return 1.0 - (z / max_distance)
def f_angle(rvec):
    # Rotación → matriz de rotación
    R, _ = cv2.Rodrigues(rvec)
    # Vector normal del marcador en coordenadas de cámara
    z_axis = R[:, 2]
    # Queremos que esté alineado con eje Z de cámara [0, 0, 1]
    cos_angle = np.dot(z_axis, np.array([0, 0, 1]))
    return max(0.0, cos_angle)  # más cos_angle → mejor
def f_size(corner_pts, image_shape):
    area = cv2.contourArea(corner_pts.astype(np.float32))
    max_area = image_shape[0] * image_shape[1]
    return min(area / (max_area * 0.05), 1.0)  # 5% del frame = confianza alta
def f_sharpness(frame, corner_pts):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corner_pts), 255)
    laplacian = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    sharpness = np.var(laplacian[mask == 255])
    return np.clip(sharpness / 100.0, 0.0, 1.0)
def f_corners_detected(corner_pts):
    if corner_pts.shape[1] == 4:
        return 1.0
    else:
        return 0.0



def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
