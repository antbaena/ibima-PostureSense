#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np

class HardPairedMosaicNode(Node):
    def __init__(self):
        super().__init__('hard_paired_mosaic_node')

        # ---- HARD-CODE ----
        # Plantilla del tópico de imagen (sin parámetros, fijo):
        image_topic_tpl = '/{}/color/image_raw/decompressed'

        # Cámaras fijas:
        CAM_00 = 'cam00/camera_00'
        CAM_01 = 'cam00/camera_01'
        CAM_02 = 'cam01/camera_02'
        CAM_03 = 'cam02/camera_03'

        # Parejas fijas (en orden de visualización):
        self.pairs = [
            ('00-01', CAM_00, CAM_01),
            ('00-02', CAM_00, CAM_02),
            ('02-03', CAM_02, CAM_03),
        ]

        # QoS de sensor (best effort, volatile)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.bridge = CvBridge()
        self.window_name = 'mosaic'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Suscriptores por cada cámara única (hardcode)
        self.subs = {
            '00': Subscriber(self, Image, image_topic_tpl.format(CAM_00), qos_profile=sensor_qos),
            '01': Subscriber(self, Image, image_topic_tpl.format(CAM_01), qos_profile=sensor_qos),
            '02': Subscriber(self, Image, image_topic_tpl.format(CAM_02), qos_profile=sensor_qos),
            '03': Subscriber(self, Image, image_topic_tpl.format(CAM_03), qos_profile=sensor_qos),
        }

        self.get_logger().info(f"Sub: {image_topic_tpl.format(CAM_00)}")
        self.get_logger().info(f"Sub: {image_topic_tpl.format(CAM_01)}")
        self.get_logger().info(f"Sub: {image_topic_tpl.format(CAM_02)}")
        self.get_logger().info(f"Sub: {image_topic_tpl.format(CAM_03)}")

        # Sincronizadores aproximados para cada pareja (hardcode)
        self.syncs = []
        self.pair_latest_img = {}  # clave: etiqueta '00-01', valor: np.ndarray

        def make_cb(label):
            def _cb(msg_a: Image, msg_b: Image):
                self.on_pair_images(label, msg_a, msg_b)
            return _cb

        # 00-01
        sync_00_01 = ApproximateTimeSynchronizer(
            [self.subs['00'], self.subs['01']], queue_size=10, slop=0.05
        )
        sync_00_01.registerCallback(make_cb('00-01'))
        self.syncs.append(sync_00_01)

        # 00-02
        sync_00_02 = ApproximateTimeSynchronizer(
            [self.subs['00'], self.subs['02']], queue_size=10, slop=0.05
        )
        sync_00_02.registerCallback(make_cb('00-02'))
        self.syncs.append(sync_00_02)

        # 02-03
        sync_02_03 = ApproximateTimeSynchronizer(
            [self.subs['02'], self.subs['03']], queue_size=10, slop=0.05
        )
        sync_02_03.registerCallback(make_cb('02-03'))
        self.syncs.append(sync_02_03)

        # Timer de render (hardcode ~15 FPS)
        self.timer = self.create_timer(1.0/15.0, self.render_mosaic)

    def on_pair_images(self, label, msg_a: Image, msg_b: Image):
        try:
            img_a = self.bridge.imgmsg_to_cv2(msg_a, desired_encoding='bgr8')
            img_b = self.bridge.imgmsg_to_cv2(msg_b, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error en pareja {label}: {e}')
            return

        # Normalizar alturas para concatenar horizontalmente
        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]
        target_h = min(ha, hb)
        if ha != target_h:
            img_a = cv2.resize(img_a, (int(wa * (target_h/ha)), target_h), interpolation=cv2.INTER_AREA)
        if hb != target_h:
            img_b = cv2.resize(img_b, (int(wb * (target_h/hb)), target_h), interpolation=cv2.INTER_AREA)

        pair_img = np.hstack((img_a, img_b))
        cv2.putText(pair_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        self.pair_latest_img[label] = pair_img

    def render_mosaic(self):
        # Orden fijo: 00-01, 00-02, 02-03
        rows = []
        for label, _, _ in self.pairs:
            img = self.pair_latest_img.get(label)
            if img is not None:
                rows.append(img)
        if not rows:
            return

        # Igualar anchos para apilar verticalmente
        min_w = min(r.shape[1] for r in rows)
        norm_rows = []
        for r in rows:
            if r.shape[1] != min_w:
                scale = min_w / r.shape[1]
                r = cv2.resize(r, (min_w, int(r.shape[0]*scale)), interpolation=cv2.INTER_AREA)
            norm_rows.append(r)
        mosaic = np.vstack(norm_rows)

        cv2.imshow(self.window_name, mosaic)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            cv2.destroyAllWindows()
            self.get_logger().info('Ventana cerrada por el usuario.')
            # rclpy.shutdown()  # opcional: dejar que el usuario mate el proceso

def main(args=None):
    rclpy.init(args=args)
    node = HardPairedMosaicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
