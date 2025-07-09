import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import ParameterType
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')
        self.get_logger().info('Aruco Detector Node initialized')

        # Declaración de parámetros con valores por defecto
        self.declare_parameter('input_topics', [''])
        self.declare_parameter('output_topics', [''])
        self.declare_parameter('dictionary_id', 5)

        # Lectura de parámetros
        input_topics = self.get_parameter('input_topics').get_parameter_value().string_array_value
        output_topics = self.get_parameter('output_topics').get_parameter_value().string_array_value
        dict_id = self.get_parameter('dictionary_id').get_parameter_value().integer_value


        # Validaciones
        if not input_topics or not output_topics:
            self.get_logger().error('input_topics y output_topics no pueden estar vacíos')
            rclpy.shutdown()
            sys.exit(1)

        if len(input_topics) != len(output_topics):
            self.get_logger().error('input_topics y output_topics deben tener igual longitud')
            rclpy.shutdown()
            sys.exit(1)

        # Inicializa OpenCV ArUco
        self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.bridge = CvBridge()

        # Crea suscripciones y publishers dinámicamente
        for in_t, out_t in zip(input_topics, output_topics):
            pub = self.create_publisher(Image, out_t, qos_profile_sensor_data)
            self.create_subscription(
                Image,
                in_t,
                lambda msg, p=pub: self.image_callback(msg, p),
                qos_profile=qos_profile_sensor_data
            )
            self.get_logger().info(f'Subscribed: {in_t} → publish annotated on {out_t}')

    def image_callback(self, msg: Image, publisher):
        # Convierte a OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Detecta marcadores
        corners, ids, _ = cv2.aruco.detectMarkers(cv_img, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cv_img, corners, ids)
        # Reconviértelo y publica
        out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out_msg.header = msg.header  # mantiene frame_id y timestamp
        publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ArucoDetectorNode()
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
