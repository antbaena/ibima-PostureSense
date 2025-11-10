#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs_py import point_cloud2
from rclpy.publisher import Publisher

# Redimensionado opcional si color y depth no coinciden en tamaño
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class DepthToPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_publisher')

        # -------- Parámetros --------
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01', 'cam01/camera_02', 'cam02/camera_03'])
        # Ajusta a tu grafo real de tópicos:
        self.declare_parameter('depth_topic_tpl', '/{}/depth/image_raw/decompressed')
        self.declare_parameter('camera_info_topic_tpl', '/{}/depth/camera_info')
        # NUEVO: color registrado/alineado a depth
        self.declare_parameter('color_topic_tpl', '/{}/color/image_raw/decompressed')
        self.declare_parameter('output_topic_tpl', '/{}/points')
        self.declare_parameter('downsample_step', 8)  # submuestreo en píxeles (>=1)
        # Frame por cámara
        self.declare_parameter('link_names', ['camera_00_color_optical_frame', 'camera_01_color_optical_frame', 'camera_02_color_optical_frame', 'camera_03_color_optical_frame'])

        self.bridge = CvBridge()

        self.cameras   = list(self.get_parameter('cameras').get_parameter_value().string_array_value)
        self.depth_tpl = self.get_parameter('depth_topic_tpl').get_parameter_value().string_value
        self.info_tpl  = self.get_parameter('camera_info_topic_tpl').get_parameter_value().string_value
        self.color_tpl = self.get_parameter('color_topic_tpl').get_parameter_value().string_value
        self.output_tpl= self.get_parameter('output_topic_tpl').get_parameter_value().string_value
        self.downsample= max(1, int(self.get_parameter('downsample_step').get_parameter_value().integer_value))
        self.link_names= list(self.get_parameter('link_names').get_parameter_value().string_array_value)

        if len(self.link_names) != len(self.cameras):
            self.get_logger().warn(
                f'link_names ({len(self.link_names)}) != cameras ({len(self.cameras)}). '
                'Se usará el frame del CameraInfo cuando no haya link_names.'
            )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # como lo tenías
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.get_logger().info(f'Cámaras: {self.cameras}')
        self.publishers_lst = {}
        self.syncs = []

        for cam in self.cameras:
            depth_topic = self.depth_tpl.format(cam)
            info_topic  = self.info_tpl.format(cam)
            color_topic = self.color_tpl.format(cam)
            out_topic   = self.output_tpl.format(cam)

            # Publisher por cámara
            pub = self.create_publisher(PointCloud2, out_topic, pub_qos)
            self.publishers_lst[cam] = pub

            # Suscriptores + sincronizador por cámara (Depth + Info + Color)
            sub_img   = Subscriber(self, Image, depth_topic, qos_profile=qos)
            sub_info  = Subscriber(self, CameraInfo, info_topic, qos_profile=qos)
            sub_color = Subscriber(self, Image, color_topic, qos_profile=qos)

            ats = ApproximateTimeSynchronizer([sub_img, sub_info, sub_color], queue_size=10, slop=0.05)  # 50 ms
            ats.registerCallback(self._make_cb(cam))
            self.syncs.append(ats)

            self.get_logger().info(
                f'[{cam}] depth: {depth_topic} | info: {info_topic} | color: {color_topic} -> pub: {out_topic}'
            )

    # ----- Callback por cámara -----
    def _make_cb(self, cam_name):
        def cb(depth_msg: Image, info_msg: CameraInfo, color_msg: Image):
            try:
                # 1) Depth a numpy (m)
                depth = self._depth_to_meters(depth_msg)  # (H, W)

                # 2) Color a RGB8 (H, W, 3)
                color = self._image_to_rgb8(color_msg)

                # Asegurar mismas dimensiones (ideal: ya alineadas)
                if depth.shape[:2] != color.shape[:2]:
                    if _HAS_CV2:
                        self.get_logger().warn(
                            f'[{cam_name}] depth {depth.shape} y color {color.shape} difieren. '
                            'Redimensionando color con NEAREST para alinear.'
                        )
                        color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        self.get_logger().error(
                            f'[{cam_name}] Dimensiones depth {depth.shape} != color {color.shape} y no hay OpenCV. '
                            'No se publica la nube.'
                        )
                        return

                # 3) Intrínsecas
                fx = info_msg.k[0]
                fy = info_msg.k[4]
                cx = info_msg.k[2]
                cy = info_msg.k[5]
                h, w = depth.shape

                step = self.downsample
                # Submuestreo
                z = depth[::step, ::step]                     # (h/step, w/step)
                rgb_sub = color[::step, ::step, :]            # (h/step, w/step, 3)

                valid = np.isfinite(z) & (z > 0.0)
                if not np.any(valid):
                    self.get_logger().warn(f'[{cam_name}] nube vacía (sin profundidad válida).')
                    return

                # malla de píxeles submuestreada
                us = np.arange(0, w, step, dtype=np.float32)
                vs = np.arange(0, h, step, dtype=np.float32)
                uu, vv = np.meshgrid(us, vs)

                uu = uu[valid]
                vv = vv[valid]
                z  = z[valid]
                rgb = rgb_sub[valid]  # (N, 3) uint8

                # 4) XYZ (metros)
                x = (uu - cx) * z / fx
                y = (vv - cy) * z / fy

                # 5) Empaquetar RGB8 en float32 (layout estándar RViz)
                # uint32 = (r<<16) | (g<<8) | b; reinterpretado como float32
                rgb = rgb.astype(np.uint32)
                rgb_uint32 = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2])
                rgb_float = rgb_uint32.view(np.float32)

                # 6) Construir PointCloud2 con campo 'rgb'
                header = Header()
                header.stamp = depth_msg.header.stamp
                header.frame_id = self._resolve_frame(cam_name, info_msg)

                fields = [
                    PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
                    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                ]

                points = np.column_stack((
                    x.astype(np.float32),
                    y.astype(np.float32),
                    z.astype(np.float32),
                    rgb_float
                ))

                pc2_msg = point_cloud2.create_cloud(header, fields, points.tolist())

                # 7) Publicar
                pb: Publisher = self.publishers_lst[cam_name]
                pb.publish(pc2_msg)

            except Exception as e:
                self.get_logger().error(f'[{cam_name}] Error generando PointCloud2 color: {e}')

        return cb

    # ----- Utilidades -----
    def _depth_to_meters(self, msg: Image) -> np.ndarray:
        """
        Convierte Image de profundidad a metros.
        - '16UC1' asumido en milímetros.
        - '32FC1' asumido en metros.
        """
        if msg.encoding in ('16UC1', 'mono16'):
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # uint16
            depth_m = cv.astype(np.float32) / 1000.0
        elif msg.encoding in ('32FC1',):
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # float32 en m
            depth_m = cv.astype(np.float32)
        else:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv.dtype == np.uint16:
                depth_m = cv.astype(np.float32) / 1000.0
            else:
                depth_m = cv.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = np.nan
        depth_m[depth_m <= 0.0] = np.nan
        return depth_m

    def _image_to_rgb8(self, msg: Image) -> np.ndarray:
        """
        Devuelve una imagen RGB8 (H, W, 3) uint8
        Acepta encodings comunes: 'rgb8', 'bgr8', 'rgba8', 'bgra8', etc.
        """
        enc = msg.encoding.lower()
        # Intentar rutas típicas
        if enc == 'rgb8':
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        elif enc == 'bgr8':
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')[:, :, ::-1]  # BGR -> RGB
        elif enc in ('rgba8', 'bgra8'):
            # Convertir a RGB (descarta alpha)
            if enc == 'rgba8':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgba8')[:, :, :3]
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')[:, :, :3][:, :, ::-1]
        else:
            # Fallback: pedir directamente rgb8 (cv_bridge hace la conversión si puede)
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        return img

    def _resolve_frame(self, cam_name: str, info_msg: CameraInfo) -> str:
        # Usa link_names[i] si existe; si no, usa frame de CameraInfo
        if cam_name in self.cameras:
            i = self.cameras.index(cam_name)
            if i < len(self.link_names) and self.link_names[i]:
                return self.link_names[i]
        return info_msg.header.frame_id if info_msg.header.frame_id else cam_name + '_optical_frame'


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
