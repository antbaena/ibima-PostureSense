#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2

def get_aruco_detector():
    """
    Devuelve un detector ArUco compatible con OpenCV 4.x,
    usando el diccionario 5x5 (250 IDs).
    """
    try:
        # OpenCV >= 4.7: API con ArucoDetector
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        # Empaquetamos un callable con la misma firma que detectMarkers
        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
        return detect, dictionary
    except AttributeError:
        # OpenCV < 4.7: API clásica
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        parameters = aruco.DetectorParameters_create()
        def detect(gray):
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
            return corners, ids
        return detect, dictionary

def depth_to_meters(depth_patch, encoding):
    """
    Convierte una pequeña ventana de profundidad a metros.
    Acepta:
      - 16UC1 => milímetros (convierte a metros)
      - 32FC1 => metros
    Hace una mediana robusta ignorando 0/NaN/inf.
    """
    d = depth_patch.astype(np.float32).flatten()
    if encoding == '16UC1':
        d = d[d > 0.0]  # 0 => sin medida
        if d.size == 0:
            return float('nan')
        return float(np.median(d)) / 1000.0
    elif encoding == '32FC1':
        d = d[np.isfinite(d) & (d > 0.0)]
        if d.size == 0:
            return float('nan')
        return float(np.median(d))
    else:
        return float('nan')

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Parámetros (puedes sobreescribirlos por CLI)
        self.declare_parameter('color_topic', '/camera_01_02/color/image_raw')
        self.declare_parameter('depth_topic', '/camera_01_02/depth/image_raw')
        self.declare_parameter('output_topic', '/aruco/overlay')
        self.declare_parameter('sync_slop', 0.10)  # tolerancia de sincronización (s)
        self.declare_parameter('depth_window', 5)  # px para estimar distancia

        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value

        # QoS de sensores (BestEffort suele ir bien con cámaras)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subs + sincronizador aproximado
        self.bridge = CvBridge()
        self.color_sub = Subscriber(self, Image, color_topic, qos_profile=sensor_qos)
        self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=sync_slop,
            allow_headerless=True
        )
        self.ts.registerCallback(self.sync_cb)

        # Publicador de imagen anotada
        self.pub = self.create_publisher(Image, output_topic, 10)

        # Detector ArUco
        self.detect_aruco, self.aruco_dictionary = get_aruco_detector()

        self.get_logger().info(
            f'Listo. Subscribiendo: {color_topic} + {depth_topic} | Publicando: {output_topic}'
        )

    def sync_cb(self, color_msg: Image, depth_msg: Image):
        try:
            # A color BGR8 (si viene mono, CvBridge lo maneja)
            frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # Profundidad: mantenemos encoding original
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth_encoding = depth_msg.encoding
        except CvBridgeError as e:
            self.get_logger().warn(f'CvBridge error: {e}')
            return

        # Detección ArUco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.detect_aruco(gray)

        if ids is not None and len(ids) > 0:
            # Dibujar contornos/IDs
            try:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            except Exception:
                # En algunas versiones drawDetectedMarkers puede fallar con ids None
                for cs, i in zip(corners, ids.flatten().tolist()):
                    pts = cs.reshape(-1, 2).astype(int)
                    for j in range(4):
                        p1 = tuple(pts[j])
                        p2 = tuple(pts[(j+1) % 4])
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cx, cy = pts.mean(axis=0).astype(int).tolist()
                    cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Distancia (m) con una pequeña ventana alrededor del centro del marcador
            win = int(self.get_parameter('depth_window').get_parameter_value().integer_value or 5)
            if win < 1:
                win = 1

            for cs, i in zip(corners, ids.flatten().tolist()):
                pts = cs.reshape(-1, 2)
                cx, cy = pts.mean(axis=0)
                cx_i, cy_i = int(round(cx)), int(round(cy))

                # recortamos una ventana válida dentro de la imagen
                x0 = max(cx_i - win//2, 0)
                y0 = max(cy_i - win//2, 0)
                x1 = min(cx_i + win//2 + 1, depth.shape[1])
                y1 = min(cy_i + win//2 + 1, depth.shape[0])

                d_m = depth_to_meters(depth[y0:y1, x0:x1], depth_encoding)
                if math.isfinite(d_m):
                    label = f'ID {i} | {d_m:.2f} m'
                else:
                    label = f'ID {i} | dist N/A'

                cv2.putText(frame, label, (x0, max(0, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Publicar imagen anotada (misma cabecera/frames que la de color)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = color_msg.header  # conserva stamp/frame_id para RViz2
            self.pub.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().warn(f'CvBridge error al publicar: {e}')


def main():
    rclpy.init()
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
