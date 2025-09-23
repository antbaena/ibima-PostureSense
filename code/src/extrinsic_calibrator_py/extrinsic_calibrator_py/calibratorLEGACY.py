from rclpy.node import Node
import rclpy
from .camera import Camera
from .utils import (
    display_camera_to_marker_table,
    display_marker_to_marker_table,
)
import time
import copy
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import random
import tf_transformations
from collections import defaultdict, deque


class ExtrinsicCalibrator(Node):
    """Calibrate extrinsic transforms between multiple cameras and markers."""

    def __init__(self):
        super().__init__("detector_aruco_node")

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # OpenCV bridge for converting ROS Image to OpenCV image
        self.bridge = CvBridge()

        # construct the cameras
        self.array_of_cameras = []

        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("marker_length", 0.05)
        self.declare_parameter("camera_names", [""])
        self.declare_parameter("image_topics", [""])
        self.declare_parameter("info_topics", [""])
        self.declare_parameter("frame_ids", [""])
        # Read parameters

        dict_name = (
            self.get_parameter("aruco_dictionary").get_parameter_value().string_value
        )
        mlen = self.get_parameter("marker_length").get_parameter_value().double_value

        camera_names = (
            self.get_parameter("camera_names").get_parameter_value().string_array_value
        )
        image_topics = (
            self.get_parameter("image_topics").get_parameter_value().string_array_value
        )
        info_topics = (
            self.get_parameter("info_topics").get_parameter_value().string_array_value
        )
        frame_ids = (
            self.get_parameter("frame_ids").get_parameter_value().string_array_value
        )

        # Check if the number of cameras is consistent
        if not (
            len(camera_names) == len(image_topics) == len(info_topics) == len(frame_ids)
            or len(camera_names) == 0
        ):
            self.get_logger().error(
                "The number of camera names, image topics, info topics and frame ids must be the same and at least 1."
            )
            self.destroy_node()
            return

        # Create a dictionary with the camera configurations
        camera_configs = []
        for i in range(len(camera_names)):
            camera_configs.append(
                {
                    "name": camera_names[i],
                    "image_topic": image_topics[i],
                    "info_topic": info_topics[i],
                    "frame_id": frame_ids[i],
                }
            )
        self.get_logger().info(f"Camera configurations: {camera_configs}")
        for idx, cam_cfg in enumerate(camera_configs):
            self.array_of_cameras.append(
                Camera(
                    node=self,
                    camera_name=cam_cfg["name"],
                    camera_id=idx,
                    image_topic=cam_cfg["image_topic"],
                    camera_info_topic=cam_cfg["info_topic"],
                    marker_length=mlen,
                    aruco_dict_name=dict_name,
                    camera_frame_id=cam_cfg["frame_id"],
                )
            )
        self.get_logger().info(
            f"Created {len(self.array_of_cameras)} cameras with the following configurations:"
        )
        for camera in self.array_of_cameras:
            camera: Camera
            self.get_logger().info(
                f"Camera {camera.camera_id}: {camera.camera_name}, image topic: {camera.image_topic}, info topic: {camera.camera_info_topic}, frame id: {camera.camera_frame_id}"
            )
        # periodically check if all cameras are calibrated
        self.timer = self.create_timer(2.0, self.check_camera_transforms_callback)

    def check_camera_transforms_callback(self):
        if all(
            [camera.are_all_transforms_precise() for camera in self.array_of_cameras]
        ):
            self.get_logger().info("All marker transforms gathered successfully")
            for camera in self.array_of_cameras:
                camera: Camera
                self.get_logger().info(
                    f"Camera {camera.camera_name} has received all marker transforms"
                )
                camera.node.destroy_subscription(camera.image_sub)
                if camera.camera_info_sub is not None:
                    # Destroy the camera_info subscription only if it exists
                    camera.node.destroy_subscription(camera.camera_info_sub)

                camera._remove_unreliable_markers()

            self.timer.cancel()
            self.initiate_calibration_routine()
            return True
        else:
            for camera in self.array_of_cameras:
                camera: Camera
                if camera.camera_matrix is None or camera.dist_coeffs is None:
                    self.get_logger().warn(
                        f"Camera {camera.camera_name} parameters not yet received. Is the camera_info topic correct?"
                    )
            self.get_logger().warn("Not all marker transforms gathered successfully")
            return False

    def initiate_calibration_routine(self):
        """Executes the calibration routine step by step, aborting if any step fails."""
        steps = [
            (self.find_max_marker_id, "Finding maximum marker ID"),

            ( 
                self.generate_is_marker_visible_from_camera_table,
                "Generating marker visibility table",
            ),


            (self.find_central_marker, "Finding central marker"),
            (
                self.generate_transform_between_markers_table,
                "Generating marker-to-marker transform table",
            ),
            (
                self.generate_path_between_markers_table,
                "Generating paths between markers",
            ),
            (self.generate_path_between_markers_with_cameras_table,
                "Generating paths between markers with cameras",
            ),
            (
                self.generate_reliable_transform_between_markers_table,
                "Generating reliable marker transforms",
            ),
            (
                self.generate_camera_to_marker_transform_table,
                "Generating camera-to-marker transform table",
            ),
            (
                self.generate_world_to_cameras_transform_table,
                "Generating world-to-cameras transform table",
            ),
            (self.broadcast_cameras_and_markers_to_world, "Broadcasting transforms"),
        ]

        for func, description in steps:
            self.get_logger().info(description)
            if not func():
                self.get_logger().error(f"Step failed: {description}")
                return False

        self.get_logger().info("Extrinsic calibration finished successfully.")
        self.get_logger().info(
            "The transforms will remain alive while this Node remains too. Hit Ctrl+C to exit"
        )

        # Keep the node alive to maintain the transforms
        try:
            while rclpy.ok():
                time.sleep(1)
        except KeyboardInterrupt:
            self.get_logger().info("Shutting down calibration node.")

    # ///////////////////////////////////////////////////////////////////////////////////////////
    def find_max_marker_id(self):
        # Find the maximum marker ID across all cameras
        self.max_marker_id = max(
            (
                marker.id
                for camera in self.array_of_cameras
                for marker in camera.markers.values()
            ),
            default=0,
        )
        self.get_logger().info(f"Max marker ID found: {self.max_marker_id}")
        return True

    def generate_is_marker_visible_from_camera_table(self):

        # Mapeo marker_id -> set(camera_id)
        marker_to_cameras = defaultdict(set)

        for camera in self.array_of_cameras:
            for marker_id in camera.markers.keys():
                marker_to_cameras[marker_id].add(camera.camera_id)

        self.get_logger().info(
            f"IDs of markers seen by cameras: {sorted(marker_to_cameras.keys())}"
        )

        num_markers = self.max_marker_id + 1
        num_cameras = len(self.array_of_cameras)
        self.is_marker_visible_from_camera_table = [
            [
                camera_id in marker_to_cameras.get(marker_id, set())
                for camera_id in range(num_cameras)
            ]
            for marker_id in range(num_markers)
        ]

        # Display the is_marker_visible_from_camera_table
        display_camera_to_marker_table(
            "Which markers does each camera see:",
            self.is_marker_visible_from_camera_table,
            self.array_of_cameras,
            self.get_logger(),
        )

        # Validaciones
        for camera in self.array_of_cameras:
            visible_markers = [
                marker_id
                for marker_id in range(num_markers)
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]
            ]
            if not visible_markers:
                self.get_logger().error(
                    f"Camera {camera.camera_name} doesn't see any marker"
                )
            elif len(visible_markers) == 1:
                self.get_logger().warn(
                    f"Camera {camera.camera_name} is only seeing one marker (Marker {visible_markers[0]})"
                )

        for marker_id in range(num_markers):
            visible_cameras = [
                camera.camera_name
                for camera in self.array_of_cameras
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]
            ]
            if not visible_cameras:
                pass  # puedes loguear si quieres
            elif len(visible_cameras) == 1:
                self.get_logger().warn(
                    f"Marker {marker_id} is only seen by one camera ({visible_cameras[0]})"
                )

        # if not any(self.is_marker_visible_from_camera_table[0]):
        #     self.get_logger().error("Marker 0 is not seen by any camera")
        #     return False

        return True

    def find_central_marker(self):
        cameras_table = [[None for _ in self.array_of_cameras] for _ in range(self.max_marker_id + 1)]
        markers_table = [[None for _ in self.array_of_cameras] for _ in range(self.max_marker_id + 1)]

        for camera in self.array_of_cameras:
            for marker_id in range(self.max_marker_id + 1):
                camera_counter = sum(self.is_marker_visible_from_camera_table[marker_id])
                marker_counter = sum(
                    self.is_marker_visible_from_camera_table[i][camera.camera_id]
                    for i in range(self.max_marker_id + 1)
                )
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    cameras_table[marker_id][camera.camera_id] = marker_counter - 1
                    markers_table[marker_id][camera.camera_id] = camera_counter - 1

        # Combinar las tablas
        scores_table = [[None for _ in self.array_of_cameras] for _ in range(self.max_marker_id + 1)]
        for camera in self.array_of_cameras:
            for marker_id in range(self.max_marker_id + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    scores_table[marker_id][camera.camera_id] = (
                        cameras_table[marker_id][camera.camera_id]
                        + markers_table[marker_id][camera.camera_id]
                    )
        scores_table

        # Verboso
        display_camera_to_marker_table(
            "Scores of each camera-marker couple:",
            scores_table,
            self.array_of_cameras,
            self.get_logger(),
        )

        self.center_marker = self.find_random_max_index(scores_table)
        self.get_logger().info(f"Our central marker is Marker {self.center_marker}")

        return True

    def generate_transform_between_markers_table(self):
        num_markers = self.max_marker_id + 1

        # Tabla global inicializada en False
        self.does_transform_exist_between_markers_table = [
            [False for _ in range(num_markers)] for _ in range(num_markers)
        ]

        for camera in self.array_of_cameras:
            camera: Camera
            camera_id = camera.camera_id
            # Obtener los marcadores visibles desde esta cámara
            visible_markers = [
                marker_id for marker_id in range(num_markers)
                if self.is_marker_visible_from_camera_table[marker_id][camera_id]
            ]

            # Inicializar tabla por cámara
            camera.can_camera_connect_two_markers_table = [
                [False for _ in range(num_markers)] for _ in range(num_markers)
            ]

            # Conectar todos los pares visibles (sin repetir y sin self-pair)
            for i in range(len(visible_markers)):
                for j in range(i + 1, len(visible_markers)):
                    m1 = visible_markers[i]
                    m2 = visible_markers[j]

                    # Marcar en tabla de la cámara
                    camera.can_camera_connect_two_markers_table[m1][m2] = True
                    camera.can_camera_connect_two_markers_table[m2][m1] = True

                    # Marcar en la tabla global
                    self.does_transform_exist_between_markers_table[m1][m2] = True
                    self.does_transform_exist_between_markers_table[m2][m1] = True

        # Mostrar resultados
        display_marker_to_marker_table(
            "Does exist a Transform between two markers through a camera:",
            self.does_transform_exist_between_markers_table,
            self.max_marker_id,
            self.get_logger(),
        )

        return True

    def generate_path_between_markers_table(self):
        num_markers = self.max_marker_id + 1

        # Inicializar la tabla de caminos entre marcadores
        self.path_between_markers_table = [[None for _ in range(num_markers)] for _ in range(num_markers)]

        for origin in range(num_markers):
            for destination in range(num_markers):
                if origin == destination:
                    self.path_between_markers_table[origin][destination] = [[]]  # Camino vacío
                    continue

                # Opcional: saltar si no hay conexión directa/indirecta
                # if not self.does_transform_exist_between_markers_table[origin][destination]:
                #     continue

                paths = self.explore_paths_between_markers(origin, destination)
                self.path_between_markers_table[origin][destination] = paths
        
        return True

    def generate_path_between_markers_with_cameras_table(self):
        num_markers = self.max_marker_id + 1

        # Generar tabla de caminos con cámaras
        self.path_between_markers_with_cameras_table = [[None for _ in range(num_markers)] for _ in range(num_markers)]

        for origin in range(num_markers):
            for destination in range(num_markers):
                paths = self.path_between_markers_table[origin][destination]

                if not paths:
                    continue

                all_camera_paths = []

                for marker_path in paths:
                    camera_paths = self.return_the_cameras_between_markers_in_path(marker_path)
                    if camera_paths:
                        all_camera_paths.extend(camera_paths)

                self.path_between_markers_with_cameras_table[origin][destination] = all_camera_paths or None

        return True

        # Check that all markers are connected with marker 0
        # TODO

    def generate_reliable_transform_between_markers_table(self):
        # Table with all the reliable transforms between markers. It tries to use as many transforms as possible in order to reduce the error
        n = self.max_marker_id + 1
        self.reliable_transform_between_markers_table = [[None for _ in range(n)] for _ in range(n)]

        for origin_marker_id in range(n):
            for destination_marker_id in range(n):
                paths = self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id]
                if paths is not None:
                    self.reliable_transform_between_markers_table[origin_marker_id][destination_marker_id] = self.compose_marker_to_marker_transform(paths)
                else:
                    self.reliable_transform_between_markers_table[origin_marker_id][destination_marker_id] = None


        # Print the reliable_transform_between_markers_table
        # self.display_marker_to_marker_table(f"Reliable transform between markers:", self.reliable_transform_between_markers_table)

        return True

    def generate_camera_to_marker_transform_table(self):
        # Create a table with the transforms between the cameras and the markers
        self.camera_to_marker_transform_table = [
            [None for _ in range(len(self.array_of_cameras))]
            for _ in range(self.max_marker_id + 1)
        ]
        for camera in self.array_of_cameras:
            camera: Camera
            for marker_id in range(self.max_marker_id + 1):
                if self.is_marker_visible_from_camera_table[marker_id][
                    camera.camera_id
                ]:
                    self.camera_to_marker_transform_table[marker_id][
                        camera.camera_id
                    ] = camera.markers[marker_id].tf
                else:
                    self.camera_to_marker_transform_table[marker_id][
                        camera.camera_id
                    ] = None

        return True

    def generate_world_to_cameras_transform_table(self):
        # Create a table with the transforms between "map" and the cameras
        self.map_to_cameras_transform_table = [
            None for _ in range(len(self.array_of_cameras))
        ]
        for camera in self.array_of_cameras:
            camera: Camera
            # check if the self.does_transform_exist_between_markers_table[camera_id]: has any True marker
            for marker_id in range(self.max_marker_id + 1):
                if self.is_marker_visible_from_camera_table[marker_id][
                    camera.camera_id
                ]:
                    self.map_to_cameras_transform_table[camera.camera_id] = (
                        self.compose_world_to_camera_transform(camera.camera_id)
                    )
                    break
                else:
                    self.map_to_cameras_transform_table[camera.camera_id] = None

        return True

    def broadcast_cameras_and_markers_to_world(self):
        # Create an array of transforms to be broadcasted
        transforms = []

        # Broadcast the transform between "marker_0" and "map"
        origin_transform = np.eye(4)

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "marker_0"
        t.child_frame_id = "map"

        translation = tf_transformations.translation_from_matrix(origin_transform)
        quaternion = tf_transformations.quaternion_from_matrix(origin_transform)

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        transforms.append(t)

        # Add all the transforms between the center_marker and the rest of the markers to the array
        for destination_marker_id in range(self.max_marker_id + 1):
            if (
                self.reliable_transform_between_markers_table[self.center_marker][
                    destination_marker_id
                ]
                is not None
            ):
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = f"marker_{self.center_marker}"
                t.child_frame_id = f"marker_{destination_marker_id}"

                transform = self.reliable_transform_between_markers_table[
                    self.center_marker
                ][destination_marker_id]
                translation = tf_transformations.translation_from_matrix(transform)
                quaternion = tf_transformations.quaternion_from_matrix(transform)

                t.transform.translation.x = translation[0]
                t.transform.translation.y = translation[1]
                t.transform.translation.z = translation[2]

                t.transform.rotation.x = -quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = -quaternion[2]
                t.transform.rotation.w = quaternion[3]

                transforms.append(t)

        # Add all the camera transforms
        for camera in self.array_of_cameras:
            camera: Camera
            if self.map_to_cameras_transform_table[camera.camera_id] is not None:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "map"
                t.child_frame_id = camera.camera_frame_id

                transform = self.map_to_cameras_transform_table[camera.camera_id]
                translation = tf_transformations.translation_from_matrix(transform)
                quaternion = tf_transformations.quaternion_from_matrix(transform)

                t.transform.translation.x = translation[0]
                t.transform.translation.y = translation[1]
                t.transform.translation.z = translation[2]
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]

                transforms.append(t)

        # Broadcast all transforms at once
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)
        return True

    # ///////////////////////////////////////////////////////////////////////////////////////////

    def explore_paths_between_markers(self, origin_marker_id, destination_marker_id):
        if origin_marker_id == destination_marker_id:
                return None

        successful_paths = []
        queue = deque()
        queue.append([origin_marker_id])  # Comienza con solo el origen

        while queue:
            current_path = queue.popleft()
            last_marker = current_path[-1]

            # Buscar hijos (marcadores conectados directamente)
            children = self.return_children(last_marker)

            for child in children:
                if child in current_path:
                    continue  # Evitar ciclos

                new_path = current_path + [child]  # NO usamos deepcopy

                if child == destination_marker_id:
                    successful_paths.append(new_path)
                else:
                    queue.append(new_path)

        return successful_paths if successful_paths else None


    def return_children(self, origin_marker):
        # Function to simply obtain the children of a marker, meaning the markers that can be reached from the origin_marker
        can_transform = []
        for child_marker in range(self.max_marker_id + 1):
            if self.does_transform_exist_between_markers_table[origin_marker][
                child_marker
            ]:
                can_transform.append(child_marker)
        return can_transform

    def return_the_cameras_between_markers_in_path(self, path):

        # Cada elemento del queue es un camino parcial (con o sin cámaras)
        queue = deque()
        queue.append(path)

        final_paths = []

        while queue:
            current_path = queue.popleft()

            # Buscar el primer par de marcadores consecutivos sin cámara intermedia
            for i in range(len(current_path) - 1):
                if isinstance(current_path[i], int) and isinstance(current_path[i + 1], int):
                    marker_a = current_path[i]
                    marker_b = current_path[i + 1]

                    for camera in self.array_of_cameras:
                        cam_id = camera.camera_id
                        if (
                            self.is_marker_visible_from_camera_table[marker_a][cam_id]
                            and self.is_marker_visible_from_camera_table[marker_b][cam_id]
                        ):
                            # Insertamos el nombre de la cámara entre ambos marcadores
                            new_path = current_path[:i+1] + [camera.camera_name] + current_path[i+1:]
                            queue.append(new_path)

                    break  # Solo expandimos un par por iteración
            else:
                # Si no hay pares [int, int], significa que ya se insertaron todas las cámaras
                final_paths.append(current_path)

        return final_paths if final_paths else None

    def compose_marker_to_marker_transform(self, paths):
        # Function to provide us with the reliable transform between two markers, trying to use as many different paths as possible in order to reduce the error
        all_paths_transforms = []
        for path in paths:
            # reverse de path
            # We are, instead, getting the reverse path so we can sue the super magic powers of the pseudoinverse later
            path.reverse()
            path_transform = np.eye(4)
            for element_index in range(len(path) - 2):
                # the elements are intercalated, between cameras and markers so we will iterate accordingly
                # check if the element_index is even (meaning it corresponds to a marker)
                if element_index % 2 == 0:
                    origin_marker_id = path[element_index]
                    camera_name = path[element_index + 1]
                    destination_marker_id = path[element_index + 2]
                    for camera in self.array_of_cameras:
                        if camera.camera_name == camera_name:
                            camera: Camera
                            if camera.can_camera_connect_two_markers_table[
                                origin_marker_id
                            ][destination_marker_id]:
                                transform = np.dot(
                                    np.linalg.inv(camera.markers[origin_marker_id].tf),
                                    camera.markers[destination_marker_id].tf,
                                )
                                path_transform = np.dot(path_transform, transform)
                            else:
                                self.get_logger().error(
                                    f"The table camera.can_camera_connect_two_markers_table of the camera {camera.camera_name} has a mistake"
                                )
                else:
                    continue
            all_paths_transforms.append(path_transform)
        stacked_transform = np.hstack(all_paths_transforms)

        # Calculate the number of 4x4 blocks in stacked_transform
        num_blocks = stacked_transform.shape[1] // 4
        stacked_identity = np.hstack([np.eye(4) for _ in range(num_blocks)])

        # Compute pseudoinverse and perform final multiplication
        pseudoinverse_result = np.linalg.pinv(stacked_transform)
        reliable_marker_to_marker_transform = np.dot(
            stacked_identity, pseudoinverse_result
        )
        return reliable_marker_to_marker_transform

    def compose_world_to_camera_transform(self, camera_id):
        # Computes the transform between "map" and the camera trying to use as many camera transforms as possible
        array_of_camera_to_world_transforms = []
        for marker_id in range(self.max_marker_id + 1):
            if self.is_marker_visible_from_camera_table[marker_id][camera_id]:
                if marker_id == 5:
                    marker_to_world_transform = np.eye(4)
                elif self.does_transform_exist_between_markers_table[marker_id][0]:
                    marker_to_world_transform = (
                        self.reliable_transform_between_markers_table[marker_id][0]
                    )
                else:
                    continue
                camera_to_marker_transform = self.camera_to_marker_transform_table[
                    marker_id
                ][camera_id]
                camera_to_world_transform = np.dot(
                    camera_to_marker_transform, marker_to_world_transform
                )

                array_of_camera_to_world_transforms.append(camera_to_world_transform)

        if not array_of_camera_to_world_transforms:
            self.get_logger().error(
                f"No valid camera-to-world transforms for camera {camera_id}. Returning identity transform."
            )
            return np.eye(4)

        stacked_transform = np.hstack(array_of_camera_to_world_transforms)
        num_blocks = stacked_transform.shape[1] // 4
        stacked_identity = np.hstack([np.eye(4) for _ in range(num_blocks)])

        pseudoinverse_result = np.linalg.pinv(stacked_transform)
        reliable_camera_transform = np.dot(stacked_identity, pseudoinverse_result)
        return reliable_camera_transform

    def find_random_max_index(self, scores_table):
        max_value = float("-inf")
        max_indices = []

        # Loop over the table to find the max value and its indices
        for marker_id in range(self.max_marker_id + 1):
            for camera in self.array_of_cameras:
                camera: Camera
                if scores_table[marker_id][camera.camera_id] is not None:
                    value = scores_table[marker_id][camera.camera_id]
                    if value > max_value:
                        max_value = value
                        # Reset list if new max is found
                        max_indices = [(marker_id, camera.camera_id)]
                    elif value == max_value:
                        # Add to list if value equals current max
                        max_indices.append((marker_id, camera.camera_id))

        # Randomly select one of the max indices
        max_marker_id, max_camera_id = random.choice(max_indices)
        if scores_table[max_marker_id][max_camera_id] == 0:
            self.get_logger().error(
                f"The center marker has no links between cameras nor markers"
            )
        return max_marker_id
