#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import os

class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('dataset_recorder')

        # Parámetros configurables
        self.declare_parameter('camera_names', ['cam00/camera_00', 'cam00/camera_01', 'cam01/camera_02', 'cam02/camera_03'])
        self.declare_parameter('topic_suffix', '/color/image_raw/compressed')
        self.declare_parameter('save_dir', os.path.expanduser('/home/ant/ibima-PostureSense/code/src/camera_postprocess/camera_postprocess/dataset'))
        self.declare_parameter('save_interval', 2.0)  # segundos
        self.declare_parameter('queue_size', 10)

        # Leer parámetros
        self.camera_names = self.get_parameter('camera_names').get_parameter_value().string_array_value
        self.topic_suffix = self.get_parameter('topic_suffix').get_parameter_value().string_value
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.save_interval = self.get_parameter('save_interval').get_parameter_value().double_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value

        # Preparar directorios
        for name in self.camera_names:
            cam_dir = os.path.join(self.save_dir, name)
            os.makedirs(cam_dir, exist_ok=True)
        self.get_logger().info(f'Directorios de guardado preparados en: {self.save_dir}')

        # Bridge y almacenamiento de última imagen
        self.bridge = CvBridge()
        # Diccionario: camera_name -> (cv_img, timestamp_str)
        self.latest = {name: (None, None) for name in self.camera_names}

        # Suscripciones a cada tópico de imagen comprimida
        # Usar sensor_data QoS (BEST_EFFORT) para compatibilidad con el driver de cámara
        for name in self.camera_names:
            topic = f"/{name}{self.topic_suffix}"
            self.create_subscription(
                CompressedImage,
                topic,
                self._make_image_cb(name),
                qos_profile_sensor_data
            )
            self.get_logger().info(f'Subscrito a {topic} como CompressedImage')

        # Timer para guardar cada intervalo
        self.create_timer(self.save_interval, self.save_images)
        self.get_logger().info(f'Guardando imágenes cada {self.save_interval}s')

    def _make_image_cb(self, name):
        def callback(msg: CompressedImage):
            # Convertir y almacenar imagen y timestamp del mensaje comprimido
            try:
                # Para mensajes CompressedImage
                cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f'Error al convertir imagen comprimida de {name}: {e}')
                return

            t = msg.header.stamp
            # Formatear timestamp: segundos.nanosegundos (9 dígitos)
            ts_str = f"{t.sec}.{t.nanosec:09d}"
            self.latest[name] = (cv_img, ts_str)
        return callback

    def save_images(self):
        # Iterar cámaras y guardar si hay imagen
        for name, (img, ts) in self.latest.items():
            if img is None or ts is None:
                continue
            cam_dir = os.path.join(self.save_dir, name)
            filename = os.path.join(cam_dir, f"{ts}.png")
            # Evitar sobrescribir
            if not os.path.exists(filename):
                cv.imwrite(filename, img)
                self.get_logger().info(f'Guardada {name}: {filename}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
