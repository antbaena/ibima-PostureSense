#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from multicam_cube_calib.srv import SaveCalibration

class CalibrationManager(Node):
    def __init__(self):
        super().__init__('calibration_manager')
        self.cli_start = self.create_client(Trigger, '/extrinsics_optimizer/start')
        self.cli_stop  = self.create_client(Trigger, '/extrinsics_optimizer/stop')
        self.cli_reset = self.create_client(Trigger, '/extrinsics_optimizer/reset')
        self.cli_save  = self.create_client(SaveCalibration, '/extrinsics_optimizer/save')
        self.get_logger().info("Servicios disponibles: start, stop, reset, save (en extrinsics_optimizer)")

        # Ejemplo: auto-arrancar tras 1s (opcional, comenta si no quieres)
        # self.create_timer(1.0, self.auto_start_once)
        self._auto_started = False

    def auto_start_once(self):
        if self._auto_started:
            return
        if self.cli_start.service_is_ready():
            self._auto_started = True
            req = Trigger.Request()
            self.cli_start.call_async(req)
            self.get_logger().info('Auto START lanzado')


def main():
    rclpy.init()
    node = CalibrationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()