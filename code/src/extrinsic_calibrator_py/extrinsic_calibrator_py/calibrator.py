#!/usr/bin/env python3
import time
import logging
from typing import Dict, Tuple, Deque
from collections import defaultdict, deque

import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node

from .aruco_cube import ArucoCube
from .msgs import DetectedMarkers  # header, camera_name, markers with .tf (4x4 np.ndarray)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ExtrinsicCalibrator(Node):
    """
    Nodo ROS2 para calibración extrínseca incremental usando un cubo perfecto de ArUco.
    """
    def __init__(self) -> None:
        super().__init__('extrinsic_calibrator')

        # Parámetros ROS2
        self.declare_parameter('marker_graph_yaml', 'config/cube.yaml')
        self.declare_parameter('detection_topics', [])
        self.declare_parameter('samples_per_pair', 20)
        self.declare_parameter('time_slop', 0.1)  # segundos

        yaml_path = self.get_parameter('marker_graph_yaml').get_parameter_value().string_value
        topics    = self.get_parameter('detection_topics').get_parameter_value().string_array_value
        self.samples_per_pair = self.get_parameter('samples_per_pair').get_parameter_value().integer_value
        self.time_slop = self.get_parameter('time_slop').get_parameter_value().double_value

        # Instanciar cubo de ArUco
        self.cube = ArucoCube.load_from_yaml(yaml_path)

        # Buffers y resultados
        self.buffers: Dict[Tuple[str,str], Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.samples_per_pair))
        self.final_tfs: Dict[Tuple[str,str], np.ndarray] = {}
        self.last_detections: Dict[str, DetectedMarkers] = {}

        # Static TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Subscripciones por cámara
        self.subs = []
        for topic in topics:
            sub = self.create_subscription(DetectedMarkers, topic, self._on_detection, 10)
            self.subs.append(sub)
        self.get_logger().info(f'Subscrito a detecciones: {topics}')

        # Servicio para disparar publicación final
        self.create_service(Empty, 'trigger_calibration', self._on_trigger_service)
        self.get_logger().info('Nodo iniciado, esperando detecciones...')

    def _on_detection(self, det: DetectedMarkers) -> None:
        """
        Recibe detecciones de una cámara de forma independiente.
        Empareja con otras detecciones recientes para formar pares.
        """
        name_i = det.camera_name
        # timestamp en segundos
        t_i = det.header.stamp.sec + det.header.stamp.nanosec * 1e-9
        self.last_detections[name_i] = (det, t_i)

        # Intentar emparejar con cada otra cámara registrada
        for name_j, (det_j, t_j) in self.last_detections.items():
            if name_j == name_i:
                continue
            # comprobar ventana de tiempo
            if abs(t_i - t_j) > self.time_slop:
                continue
            # procesar par
            self._process_pair(det, det_j)

    def _process_pair(self, det_i: DetectedMarkers, det_j: DetectedMarkers) -> None:
        cam_i, cam_j = det_i.camera_name, det_j.camera_name
        key = tuple(sorted((cam_i, cam_j)))
        if key in self.final_tfs:
            return  # ya calculado

        # buscar marcadores comunes o vía cubo
        for mi in det_i.markers:
            for mj in det_j.markers:
                if mi.id == mj.id:
                    T = mi.tf @ np.linalg.inv(mj.tf)
                else:
                    try:
                        T_c = self.cube.get_transform(mi.id, mj.id)
                    except KeyError:
                        continue
                    T = mi.tf @ T_c @ np.linalg.inv(mj.tf)

                # acumular
                buf = self.buffers[key]
                buf.append(T)
                count = len(buf)
                self.get_logger().debug(f'[{key}] muestras: {count}/{self.samples_per_pair}')
                if count >= self.samples_per_pair:
                    mean_T = self._average(buf)
                    self.final_tfs[key] = mean_T
                    self.get_logger().info(f'TF final para {key} calculada con {count} muestras')
                    self._publish_static_tf(key, mean_T)
                    return

    def _average(self, Ts: Deque[np.ndarray]) -> np.ndarray:
        """Promedia traslaciones y quaterniones de lista de Ts homogéneas."""
        trans = np.stack([T[:3,3] for T in Ts], axis=0)
        mean_t = trans.mean(axis=0)
        quats = [tf2_ros.transformations.quaternion_from_matrix(T) for T in Ts]
        Q = np.array(quats)
        q = Q.sum(axis=0)
        q /= np.linalg.norm(q)
        M = tf2_ros.transformations.quaternion_matrix(q)
        M[:3,3] = mean_t
        return M

    def _publish_static_tf(self, key: Tuple[str,str], M: np.ndarray) -> None:
        a, b = key
        t = M[:3,3]
        q = tf2_ros.transformations.quaternion_from_matrix(M)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = a
        msg.child_frame_id = b
        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = float(t[2])
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(msg)

    def _on_trigger_service(self, request, response):
        """Publica todas las TF calculadas hasta el momento."""
        self.get_logger().info('Trigger recibido: republicando todas las TF finales')
        for key, M in self.final_tfs.items():
            self._publish_static_tf(key, M)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
