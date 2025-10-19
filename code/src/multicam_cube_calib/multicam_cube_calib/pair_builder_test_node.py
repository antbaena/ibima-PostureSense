#!/usr/bin/env python3
import rclpy
import numpy as np
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from multicam_cube_calib_interfaces.msg import CamerasMarkersList
from multicam_cube_calib.se3 import tf_to_mat, mat_to_tf


class PairTFPublisher(Node):
    """Publica transformaciones TF entre cámaras y la pose del cubo.

    - Lee detecciones sincronizadas `CamerasMarkersList`.
    - Usa un `cube_config` YAML con la pose de cada marcador en el frame del cubo (marker -> cube).
    - Publica:
        * TF cam_i -> cam_j para cada par de cámaras visibles.
        * TF cam_best -> cube_frame, donde cam_best es la cámara con mayor confianza.
    """

    def __init__(self):
        super().__init__('pair_tf_publisher')

        # Parámetros
        self.declare_parameter('markers_topic', '/camera_markers_async')
        self.declare_parameter('cube_config', '/home/mapir/ibima-PostureSense/code/src/multicam_cube_calib/config/cube.yaml')
        self.declare_parameter('cube_frame', 'cube')            # nombre del frame del cubo
        self.declare_parameter('publish_cube', True)            # habilitar publicación del cubo
        self.declare_parameter('log_level_debug', False)        # para activar logs debug

        self.markers_topic = self.get_parameter('markers_topic').get_parameter_value().string_value
        self.cube_config_path = self.get_parameter('cube_config').get_parameter_value().string_value
        self.cube_frame = self.get_parameter('cube_frame').get_parameter_value().string_value
        self.publish_cube = self.get_parameter('publish_cube').get_parameter_value().bool_value
        self.log_debug = self.get_parameter('log_level_debug').get_parameter_value().bool_value

        self.qos = QoSProfile(depth=10)

        # Cargar configuración del cubo (poses de marcadores)
        self.marker_to_cube = {}  # id -> 4x4 numpy matrix
        if self.cube_config_path:
            try:
                with open(self.cube_config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                for mid, v in (cfg.get('markers', {}) or {}).items():
                    try:
                        mid_i = int(mid)
                        if 'T' in v:
                            T = np.array(v['T'], dtype=float)
                            if T.shape == (4, 4):
                                self.marker_to_cube[mid_i] = T
                        elif 't' in v and 'q' in v:
                            t = np.array(v['t'], dtype=float)
                            q = np.array(v['q'], dtype=float)  # [x,y,z,w]
                            T = np.eye(4)
                            x, y, z, w = q
                            R = np.array([
                                [1-2*(y*y+z*z), 2*(x*y- z*w), 2*(x*z+ y*w)],
                                [2*(x*y+ z*w), 1-2*(x*x+z*z), 2*(y*z- x*w)],
                                [2*(x*z- y*w), 2*(y*z+ x*w), 1-2*(x*x+y*y)]
                            ], dtype=float)
                            T[:3, :3] = R
                            T[:3, 3] = t
                            self.marker_to_cube[mid_i] = T
                    except Exception:
                        self.get_logger().warning(f"Formato inválido para marker {mid} en cube_config; ignorando")
                self.get_logger().info(f"Cargadas {len(self.marker_to_cube)} poses de marcadores desde {self.cube_config_path}")
            except Exception as e:
                self.get_logger().warning(f"No se pudo cargar cube_config '{self.cube_config_path}': {e}")
        else:
            self.get_logger().warning("No se proporcionó parámetro 'cube_config'; no se podrán enlazar marcadores con el cubo")

        # TF broadcaster y suscripción
        self.tf_broadcaster = TransformBroadcaster(self)
        self.sub = self.create_subscription(CamerasMarkersList, self.markers_topic, self.on_markers, self.qos)
        self.get_logger().info(f"Suscrito a {self.markers_topic}, publicando TFs cam_i -> cam_j y {self.cube_frame}")

    def on_markers(self, msg: CamerasMarkersList):
        # Requiere ≥ 2 cámaras con detecciones para pares; ≥1 para el cubo
        cams_with_dets = [cm for cm in msg.cameras if len(cm.marker_ids) > 0]
        if len(cams_with_dets) == 0:
            if self.log_debug:
                self.get_logger().debug("Ninguna cámara con detecciones, ignorando")
            return

        # Construir mapa cam -> T_cam_to_cube usando el mejor marcador
        cam_to_cube = {}
        cam_conf = {}

        for cam_msg in cams_with_dets:
            cam = cam_msg.camera  # nombre del frame de la cámara
            best_score = -1.0
            best_T = None
            best_mid = None
            for mid, tform, conf in zip(cam_msg.marker_ids, cam_msg.t_cam_marker, cam_msg.confidence):
                if mid not in self.marker_to_cube:
                    continue
                try:
                    t_cam_marker = tf_to_mat(tform)
                    T_marker_to_cube = self.marker_to_cube[mid]
                    t_cam_to_cube = t_cam_marker @ T_marker_to_cube
                except Exception as e:
                    self.get_logger().warning(f"Error calculando T para cam {cam} marker {mid}: {e}")
                    continue
                if conf > best_score:
                    best_score = conf
                    best_T = t_cam_to_cube
                    best_mid = mid

            if best_T is not None:
                cam_to_cube[cam] = best_T
                cam_conf[cam] = float(max(1e-6, best_score))
                if self.log_debug:
                    self.get_logger().debug(f"Cámara {cam}: mejor marker {best_mid} conf={best_score:.4f}")

        # --- Publicar TF del cubo (cam_best -> cube_frame) ---
        if self.publish_cube and len(cam_to_cube) > 0:
            # elige la cámara con mayor confianza
            cam_best = max(cam_conf.items(), key=lambda kv: kv[1])[0]
            T_best = cam_to_cube[cam_best]

            ts_cube = TransformStamped()
            ts_cube.header = Header()
            ts_cube.header.stamp = msg.header.stamp if hasattr(msg, 'header') else self.get_clock().now().to_msg()
            ts_cube.header.frame_id = cam_best
            ts_cube.child_frame_id = self.cube_frame
            ts_cube.transform = mat_to_tf(T_best)
            self.tf_broadcaster.sendTransform(ts_cube)

            if self.log_debug:
                self.get_logger().debug(
                    f"Publicado cubo: {cam_best} -> {self.cube_frame} (conf={cam_conf[cam_best]:.4f})"
                )

        # --- Publicar TFs entre cámaras (requiere ≥2) ---
        cams = list(cam_to_cube.keys())
        if len(cams) < 2:
            return

        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                ci, cj = cams[i], cams[j]
                Ti = cam_to_cube[ci]
                Tj = cam_to_cube[cj]

                try:
                    Tij = Ti @ np.linalg.inv(Tj)  # i -> j
                except np.linalg.LinAlgError as e:
                    self.get_logger().warning(f"No se pudo invertir T de {cj}: {e}")
                    continue

                ts = TransformStamped()
                ts.header = Header()
                ts.header.stamp = msg.header.stamp if hasattr(msg, 'header') else self.get_clock().now().to_msg()
                ts.header.frame_id = ci
                ts.child_frame_id = cj
                ts.transform = mat_to_tf(Tij)
                self.tf_broadcaster.sendTransform(ts)

                if self.log_debug:
                    self.get_logger().debug(
                        f"TF {ci} -> {cj} | dist = {np.linalg.norm(Tij[:3, 3]) * 100:.2f} cm"
                    )


def main():
    rclpy.init()
    node = PairTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
