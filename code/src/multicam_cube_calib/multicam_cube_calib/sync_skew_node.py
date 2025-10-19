#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SyncSkewNode(Node):
    def __init__(self):
        super().__init__('sync_skew_node')

        # Parámetros
        self.declare_parameter('cameras', ['cam00/camera_00', 'cam00/camera_01', 'cam01/camera_02', 'cam02/camera_03'])
        self.declare_parameter('image_topic_tpl', '/{}/color/image_raw/decompressed')
        self.declare_parameter('queue_size', 20)   # cola para el sincronizador
        self.declare_parameter('slop', 999.0)       # tolerancia (s) para sincronización aproximada
        self.declare_parameter('exact', False)     # si prefieres exacta, pon True (y reduce slop)

        cameras = self.get_parameter('cameras').get_parameter_value().string_array_value
        topic_tpl = self.get_parameter('image_topic_tpl').get_parameter_value().string_value
        queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        slop = float(self.get_parameter('slop').get_parameter_value().double_value or 0.05)
        exact = self.get_parameter('exact').get_parameter_value().bool_value

        topics = [topic_tpl.format(name) for name in cameras]
        if len(topics) < 2:
            self.get_logger().error('Necesitas al menos 2 topics.')
            rclpy.shutdown()
            return

        # QoS típico de sensores
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Subscriptores de message_filters
        self.subs = [Subscriber(self, Image, t, qos_profile=qos) for t in topics]

        # Sincronizador (aproximado por defecto)
        if exact:
            from message_filters import TimeSynchronizer
            self.sync = TimeSynchronizer(self.subs, queue_size)
        else:
            self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=queue_size, slop=slop, allow_headerless=False)

        self.sync.registerCallback(self.synced_cb)

        self.get_logger().info('Escuchando:\n  ' + '\n  '.join(topics))
        self.get_logger().info(f'Modo: {"EXACT" if exact else "APPROX"} | queue_size={queue_size} | slop={slop:.3f}s')

    def synced_cb(self, *msgs: Image):
        # Convierte stamps a segundos (float) relativos al epoch ROS
        stamps = [Time.from_msg(m.header.stamp).nanoseconds / 1e9 for m in msgs]
        # Desincronización (máx - mín) en ms
        skew_ms = (max(stamps) - min(stamps)) * 1000.0

        # Info extra: offsets relativos al 1º mensaje
        t0 = stamps[0]
        rel_ms = [(t - t0) * 1000.0 for t in stamps]

        # Construye una línea compacta
        cams = self.get_parameter('cameras').get_parameter_value().string_array_value
        pairs = ', '.join([f'{cams[i]}: {rel_ms[i]:+.2f} ms' for i in range(len(msgs))])

        self.get_logger().info(f'Desincronización = {skew_ms:.2f} ms | offsets: {pairs}')

def main():
    rclpy.init()
    node = SyncSkewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
