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


class ExtrinsicCalibrator(Node):
    """Calibrate extrinsic transforms between multiple cameras and markers."""

    def __init__(self):
        super().__init__('detector_aruco_node')

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # OpenCV bridge for converting ROS Image to OpenCV image
        self.bridge = CvBridge()

        # construct the cameras
        self.array_of_cameras = []

        self.declare_parameter('aruco_dictionary', 'DICT_6X6_250')
        self.declare_parameter('marker_length', 0.05)
        self.declare_parameter('camera_names', [''])
        self.declare_parameter('image_topics', [''])
        self.declare_parameter('info_topics', [''])
        self.declare_parameter('frame_ids', [''])
        # Read parameters

        dict_name = self.get_parameter(
            'aruco_dictionary').get_parameter_value().string_value
        mlen = self.get_parameter(
            'marker_length').get_parameter_value().double_value

        camera_names = self.get_parameter(
            'camera_names').get_parameter_value().string_array_value
        image_topics = self.get_parameter(
            'image_topics').get_parameter_value().string_array_value
        info_topics = self.get_parameter(
            'info_topics').get_parameter_value().string_array_value
        frame_ids = self.get_parameter(
            'frame_ids').get_parameter_value().string_array_value

        # Check if the number of cameras is consistent
        if not (len(camera_names) == len(image_topics) == len(info_topics) == len(frame_ids) or
                len(camera_names) == 0):
            self.get_logger().error(
                "The number of camera names, image topics, info topics and frame ids must be the same and at least 1.")
            self.destroy_node()
            return

        # Create a dictionary with the camera configurations
        camera_configs = []
        for i in range(len(camera_names)):
            camera_configs.append({
                'name': camera_names[i],
                'image_topic': image_topics[i],
                'info_topic': info_topics[i],
                'frame_id': frame_ids[i]
            })
        self.get_logger().info(f"Camera configurations: {camera_configs}")
        for idx, cam_cfg in enumerate(camera_configs):
            self.array_of_cameras.append(Camera(
                node=self,
                camera_name=cam_cfg['name'],
                camera_id=idx,
                image_topic=cam_cfg['image_topic'],
                camera_info_topic=cam_cfg['info_topic'],
                marker_length=mlen,
                aruco_dict_name=dict_name,
                camera_frame_id=cam_cfg['frame_id'],
            ))
        self.get_logger().info(
            f"Created {len(self.array_of_cameras)} cameras with the following configurations:")
        for camera in self.array_of_cameras:
            camera: Camera
            self.get_logger().info(
                f"Camera {camera.camera_id}: {camera.camera_name}, image topic: {camera.image_topic}, info topic: {camera.camera_info_topic}, frame id: {camera.camera_frame_id}")
        # periodically check if all cameras are calibrated
        self.timer = self.create_timer(
            5.0, self.check_camera_transforms_callback)

    def check_camera_transforms_callback(self):
        if all([camera.are_all_transforms_precise() for camera in self.array_of_cameras]):
            self.get_logger().info("All marker transforms gathered successfully")
            for camera in self.array_of_cameras:
                camera: Camera
                self.get_logger().info(
                    f"Camera {camera.camera_name} has received all marker transforms")
                camera.image_sub.destroy()
                camera.camera_info_sub.destroy()
            self.timer.cancel()
            self.initiate_calibration_routine()
            return True
        else:
            for camera in self.array_of_cameras:
                camera: Camera
                if camera.camera_matrix is None or camera.dist_coeffs is None:
                    self.get_logger().warn(
                        f"Camera {camera.camera_name} parameters not yet received. Is the camera_info topic correct?")
            self.get_logger().warn("Not all marker transforms gathered successfully")
            return False

    def initiate_calibration_routine(self):
        """Executes the calibration routine step by step, aborting if any step fails."""
        steps = [
            (self.generate_is_marker_visible_from_camera_table,
             "Generating marker visibility table"),
            (self.find_central_marker, "Finding central marker"),
            (self.generate_transform_between_markers_table,
             "Generating marker-to-marker transform table"),
            (self.generate_path_between_markers_table,
             "Generating paths between markers"),
            (self.generate_reliable_transform_between_markers_table,
             "Generating reliable marker transforms"),
            (self.generate_camera_to_marker_transform_table,
             "Generating camera-to-marker transform table"),
            (self.generate_world_to_cameras_transform_table,
             "Generating world-to-cameras transform table"),
            (self.broadcast_cameras_and_markers_to_world, "Broadcasting transforms"),
        ]

        for func, description in steps:
            self.get_logger().info(description)
            if not func():
                self.get_logger().error(f"Step failed: {description}")
                return False

        self.get_logger().info("Extrinsic calibration finished successfully.")
        self.get_logger().info(
            "The transforms will remain alive while this Node remains too. Hit Ctrl+C to exit")

        # Keep the node alive to maintain the transforms
        try:
            while rclpy.ok():
                time.sleep(1)
        except KeyboardInterrupt:
            self.get_logger().info("Shutting down calibration node.")


# ///////////////////////////////////////////////////////////////////////////////////////////

    def generate_is_marker_visible_from_camera_table(self):
        # Build a 2D array where we can see which markers are visible by each camera is_marker_visible_from_camera_table[camera_id][marker_id]
        self.largest_marker = max(
            (max(camera.markers, default=0)
             for camera in self.array_of_cameras if camera.markers),
            default=0
        )

        self.get_logger().info(f"Largest marker: {self.largest_marker}")
        self.is_marker_visible_from_camera_table = [[False for _ in range(
            len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]

        # Get array of all seen markers
        seen_markers = set()
        for camera in self.array_of_cameras:
            seen_markers.update(camera.markers.keys())

        # Fill is_marker_visible_from_camera_table
        for camera in self.array_of_cameras:
            marker_ids_in_camera = camera.markers.keys() if camera.markers else []
            for marker_id in seen_markers:
                if marker_id in marker_ids_in_camera:
                    self.get_logger().info(
                        f"Camera {camera.camera_name} sees marker {marker_id}")
                    self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] = True

        # Display the is_marker_visible_from_camera_table
        display_camera_to_marker_table("Which markers does each camera see:",
                                       self.is_marker_visible_from_camera_table, self.array_of_cameras, self.get_logger())

        # Check for exceptional cases
        for camera in self.array_of_cameras:
            # Check if a camera is not seeing any marker
            if all([not self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for marker_id in range(self.largest_marker + 1)]):
                self.get_logger().error(
                    f"Camera {self.array_of_cameras[camera.camera_id].camera_name} doesn't see any marker")
            # Check if a camera is only seeing one marker
            elif sum(self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for marker_id in range(self.largest_marker + 1)) == 1:
                if (self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for marker_id in range(self.largest_marker + 1)):
                    self.get_logger().warn(
                        f"Camera {camera.camera_name} is only seeing one marker (Marker {marker_id})")

        # Check if any row has all False values
        for marker_id in range(self.largest_marker + 1):
            # Check if a marker is not seen by any camera
            if all([not self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for camera in self.array_of_cameras]):
                # self.get_logger().warn(f"Marker {marker_id} is not seen by any camera")
                pass
            # Check if a marker is olny seen by one camera
            elif sum(self.is_marker_visible_from_camera_table[marker_id][camera.camera_id] for camera in self.array_of_cameras) == 1:
                for camera in self.array_of_cameras:
                    camera: Camera
                    if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                        self.get_logger().warn(
                            f"Marker {marker_id} is only seen by one camera (Camera {camera.camera_name})")

        # Check if specifically marker 0 is seen by any camera
        if not any(self.is_marker_visible_from_camera_table[0][camera.camera_id] for camera in self.array_of_cameras):
            self.get_logger().error(f"Marker 0 is not seen by any camera")
            return False

        else:
            return True

    def find_central_marker(self):
        # Table with the camera scores
        cameras_table = [[None for _ in range(
            len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        # Table with the marker scores
        markers_table = [[None for _ in range(
            len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        self.scores_table = [[None for _ in range(
            len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]

        # For each cell in the table, provide two tables, one indicating how many other markers are seen byt the same camera, and another indicating how many cameras sees one individual camera
        for camera in self.array_of_cameras:
            camera: Camera
            for marker_id in range(self.largest_marker + 1):
                camera_counter = 0
                marker_counter = 0
                for i in range(len(self.is_marker_visible_from_camera_table[marker_id])):
                    if self.is_marker_visible_from_camera_table[marker_id][i]:
                        camera_counter += 1
                for i in range(len(self.is_marker_visible_from_camera_table)):
                    if self.is_marker_visible_from_camera_table[i][camera.camera_id]:
                        marker_counter += 1
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    cameras_table[marker_id][camera.camera_id] = marker_counter-1
                    markers_table[marker_id][camera.camera_id] = camera_counter-1

        # The total result will be the addition of the two tables
        for camera in self.array_of_cameras:
            camera: Camera
            for marker_id in range(self.largest_marker + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    self.scores_table[marker_id][camera.camera_id] = cameras_table[marker_id][camera.camera_id] + \
                        markers_table[marker_id][camera.camera_id]
                else:
                    self.scores_table[marker_id][camera.camera_id] = None

        # Display the scores_table
        display_camera_to_marker_table(
            "Scores of each camera-marker couple:", self.scores_table, self.array_of_cameras, self.get_logger())

        # Check which marker has better punctuation by checking the maximum value of the table and returning its indices
        self.center_marker = self.find_random_max_index(self.scores_table)
        self.get_logger().info(
            f"Our central marker is Marker {self.center_marker}")

        return True

    def generate_transform_between_markers_table(self):
        # Table with all the transforms we know between markers due to the camera transforms

        # Create one table connected markers by a single camera
        for camera in self.array_of_cameras:
            camera: Camera
            camera.can_camera_connect_two_markers_table = [[False for _ in range(
                self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
            for origin_marker_id in range(self.largest_marker + 1):
                for destination_marker_id in range(self.largest_marker + 1):
                    if origin_marker_id != destination_marker_id:
                        if self.is_marker_visible_from_camera_table[origin_marker_id][camera.camera_id] and self.is_marker_visible_from_camera_table[destination_marker_id][camera.camera_id]:
                            camera.can_camera_connect_two_markers_table[
                                origin_marker_id][destination_marker_id] = True

        # Merge the previous table into a huge table to check which markers can be connected at all
        self.does_transform_exist_between_markers_table = [[False for _ in range(
            self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for camera in self.array_of_cameras:
            camera: Camera
            for origin_marker_id in range(self.largest_marker + 1):
                for destination_marker_id in range(self.largest_marker + 1):
                    if camera.can_camera_connect_two_markers_table[origin_marker_id][destination_marker_id] == True:
                        self.does_transform_exist_between_markers_table[
                            origin_marker_id][destination_marker_id] = True

        # Print the does_transform_exist_between_markers_table
        display_marker_to_marker_table(f"Does exist a Transform between two markers through a camera:",
                                       self.does_transform_exist_between_markers_table, self.largest_marker, self.get_logger())

        return True

    def generate_path_between_markers_table(self):

        # Table with all the different combination of paths to go from one marker to another
        self.path_between_markers_table = [[None for _ in range(
            self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for origin_marker_id in range(self.largest_marker + 1):
            for destination_marker_id in range(self.largest_marker + 1):
                self.path_between_markers_table[origin_marker_id][destination_marker_id] = self.explore_paths_between_markers(
                    origin_marker_id, destination_marker_id)

        # Print the path_between_markers_table
        # self.display_marker_to_marker_table(f"Path between markers:", self.path_between_markers_table)

        # Generate the even more expanded table of paths
        self.path_between_markers_with_cameras_table = [[[] for _ in range(
            self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for origin_marker_id in range(self.largest_marker + 1):
            for destination_marker_id in range(self.largest_marker + 1):
                if self.path_between_markers_table[origin_marker_id][destination_marker_id] is not None:
                    for path in self.path_between_markers_table[origin_marker_id][destination_marker_id]:
                        new_paths = self.return_the_cameras_between_markers_in_path(
                            path)
                        if new_paths is not None:
                            for path in new_paths:
                                self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id].append(
                                    path)
                else:
                    self.path_between_markers_with_cameras_table[
                        origin_marker_id][destination_marker_id] = None

        # Print the path_between_markers_with_cameras_table
        # self.display_marker_to_marker_table(f"Path between markers including cameras:", self.path_between_markers_with_cameras_table)

        return True
        # Check that all markers are connected with marker 0
        # TODO

    def generate_reliable_transform_between_markers_table(self):
        # Table with all the reliable transforms between markers. It tries to use as many transforms as possible in order to reduce the error
        self.reliable_transform_between_markers_table = [[None for _ in range(
            self.largest_marker + 1)] for _ in range(self.largest_marker + 1)]
        for origin_marker_id in range(self.largest_marker + 1):
            for destination_marker_id in range(self.largest_marker + 1):
                if self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id] == None:
                    self.reliable_transform_between_markers_table[
                        origin_marker_id][destination_marker_id] = None
                else:
                    self.reliable_transform_between_markers_table[origin_marker_id][destination_marker_id] = self.compose_marker_to_marker_transform(
                        self.path_between_markers_with_cameras_table[origin_marker_id][destination_marker_id])

        # Print the reliable_transform_between_markers_table
        # self.display_marker_to_marker_table(f"Reliable transform between markers:", self.reliable_transform_between_markers_table)

        return True

    def generate_camera_to_marker_transform_table(self):
        # Create a table with the transforms between the cameras and the markers
        self.camera_to_marker_transform_table = [[None for _ in range(
            len(self.array_of_cameras))] for _ in range(self.largest_marker + 1)]
        for camera in self.array_of_cameras:
            camera: Camera
            for marker_id in range(self.largest_marker + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    self.camera_to_marker_transform_table[marker_id][
                        camera.camera_id] = camera.markers[marker_id].reliable_transform
                else:
                    self.camera_to_marker_transform_table[marker_id][camera.camera_id] = None

        return True

    def generate_world_to_cameras_transform_table(self):
        # Create a table with the transforms between "map" and the cameras
        self.map_to_cameras_transform_table = [
            None for _ in range(len(self.array_of_cameras))]
        for camera in self.array_of_cameras:
            camera: Camera
            # check if the self.does_transform_exist_between_markers_table[camera_id]: has any True marker
            for marker_id in range(self.largest_marker + 1):
                if self.is_marker_visible_from_camera_table[marker_id][camera.camera_id]:
                    self.map_to_cameras_transform_table[camera.camera_id] = self.compose_world_to_camera_transform(
                        camera.camera_id)
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

        translation = tf_transformations.translation_from_matrix(
            origin_transform)
        quaternion = tf_transformations.quaternion_from_matrix(
            origin_transform)

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        transforms.append(t)

        # Add all the transforms between the center_marker and the rest of the markers to the array
        for destination_marker_id in range(self.largest_marker + 1):
            if self.reliable_transform_between_markers_table[self.center_marker][destination_marker_id] is not None:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = f"marker_{self.center_marker}"
                t.child_frame_id = f"marker_{destination_marker_id}"

                transform = self.reliable_transform_between_markers_table[
                    self.center_marker][destination_marker_id]
                translation = tf_transformations.translation_from_matrix(
                    transform)
                quaternion = tf_transformations.quaternion_from_matrix(
                    transform)

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
                translation = tf_transformations.translation_from_matrix(
                    transform)
                quaternion = tf_transformations.quaternion_from_matrix(
                    transform)

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
        # Function to retrieve all the paths between markers
        if origin_marker_id == destination_marker_id:
            return None
        else:
            # Temporary starting path with the origin_marker_id
            array_of_paths = []
            current_path = []
            current_path.append(origin_marker_id)
            array_of_paths.append(current_path)
            return self.evaluate_paths(origin_marker_id, destination_marker_id, array_of_paths)

    def evaluate_paths(self, origin_marker_id, destination_marker_id, current_array_of_paths):
        # Array of paths that successfully go from origin_marker_id to destination_marker_id
        array_of_successful_paths = []
        # Temporary array of paths that are still candidates to be added to the array_of_successful_paths
        array_of_paths = []
        array_of_paths = copy.deepcopy(current_array_of_paths)
        # For each path in the array of paths, find the children of the last element of the path. For each child, duplicate the path, adding the child at the end
        temp_array_of_paths = []
        for path in array_of_paths:
            children = self.return_children(path[-1])
            for child in children:
                new_path = copy.deepcopy(path)
                new_path.append(child)
                temp_array_of_paths.append(new_path)
        # Temporary array of paths that are still candidates to be added to the array_of_successful_paths, now with the children added
        array_of_paths = copy.deepcopy(temp_array_of_paths)

        # Check if any of the paths can be added to the array of finished paths
        temp_array_of_paths = copy.deepcopy(array_of_paths)
        for path in temp_array_of_paths:
            # if the newly added child is the origin, pop the path out of the candidates array
            if path[-1] == origin_marker_id:
                array_of_paths.remove(path)
            # if the newly added child was already in the path, pop the path out the candidates array
            elif path[-1] in path[:-1]:
                array_of_paths.remove(path)
            # if the newly added child is the destination, add it to the array_of_successful_paths
            elif destination_marker_id == path[-1]:
                if path not in array_of_successful_paths:
                    array_of_successful_paths.append(path)
                array_of_paths.remove(path)

        # If there are still candidates in the array_of_paths iterate recursively on those
        if len(array_of_paths) > 0:
            new_good_paths = self.evaluate_paths(
                origin_marker_id, destination_marker_id, array_of_paths)
            if new_good_paths is not None:
                for new_path in new_good_paths:
                    if new_path not in array_of_successful_paths:
                        array_of_successful_paths.append(new_path)

        # If no path was found return None
        if array_of_successful_paths is None:
            return None
        elif len(array_of_successful_paths) == 0:
            return None
        else:
            return array_of_successful_paths

    def return_children(self, origin_marker):
        # Function to simply obtain the children of a marker, meaning the markers that can be reached from the origin_marker
        can_transform = []
        for child_marker in range(self.largest_marker + 1):
            if self.does_transform_exist_between_markers_table[origin_marker][child_marker]:
                can_transform.append(child_marker)
        return can_transform

    def return_the_cameras_between_markers_in_path(self, path):
        # Function to insert the cameras in the paths, needed to transform from anoe to another
        final_array_of_paths = []

        paths_to_evaluate = []
        temp_path = copy.deepcopy(path)
        paths_to_evaluate.append(temp_path)

        for path_being_evaluated in paths_to_evaluate:
            # Check if the path has been evaluated by checking if the last element is a number, and the previous a string
            if type(path_being_evaluated[-2]) is str and type(path_being_evaluated[-1]) is int:
                # If it is, then the path has been evaluated
                if path_being_evaluated not in final_array_of_paths:
                    final_array_of_paths.append(path_being_evaluated)
                paths_to_evaluate.pop(
                    paths_to_evaluate.index(path_being_evaluated))
            else:
                # If it is not, then the path has not been evaluated
                # Find the first element in the path which next element is not a string
                for marker_id_index in range(len(path_being_evaluated)-1):
                    if type(path_being_evaluated[marker_id_index]) is int and type(path_being_evaluated[marker_id_index + 1]) is not str:
                        for camera in self.array_of_cameras:
                            camera: Camera
                            if self.is_marker_visible_from_camera_table[path_being_evaluated[marker_id_index]][camera.camera_id] and self.is_marker_visible_from_camera_table[path_being_evaluated[marker_id_index + 1]][camera.camera_id]:
                                new_path = copy.deepcopy(path_being_evaluated)
                                new_path.insert(
                                    marker_id_index + 1, camera.camera_name)
                                paths_to_evaluate.append(new_path)

                        break
                    else:
                        continue

        if len(final_array_of_paths) > 0:
            return final_array_of_paths
        else:
            return None

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
                    camera_name = path[element_index+1]
                    destination_marker_id = path[element_index + 2]
                    for camera in self.array_of_cameras:
                        if camera.camera_name == camera_name:
                            camera: Camera
                            if camera.can_camera_connect_two_markers_table[origin_marker_id][destination_marker_id]:
                                transform = np.dot(np.linalg.inv(camera.markers[origin_marker_id].reliable_transform),
                                                   camera.markers[destination_marker_id].reliable_transform)
                                path_transform = np.dot(
                                    path_transform, transform)
                            else:
                                self.get_logger().error(
                                    f"The table camera.can_camera_connect_two_markers_table of the camera {camera.camera_name} has a mistake")
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
            stacked_identity, pseudoinverse_result)
        return reliable_marker_to_marker_transform

    def compose_world_to_camera_transform(self, camera_id):
        # Computes the transform between "map" and the camera trying to use as many camera transforms as possible
        array_of_camera_to_world_transforms = []
        for marker_id in range(self.largest_marker + 1):
            if self.is_marker_visible_from_camera_table[marker_id][camera_id]:
                if marker_id == 0:
                    marker_to_world_transform = np.eye(4)
                elif self.does_transform_exist_between_markers_table[marker_id][0]:
                    marker_to_world_transform = self.reliable_transform_between_markers_table[
                        marker_id][0]
                else:
                    continue
                camera_to_marker_transform = self.camera_to_marker_transform_table[
                    marker_id][camera_id]
                camera_to_world_transform = np.dot(
                    camera_to_marker_transform, marker_to_world_transform)

                array_of_camera_to_world_transforms.append(
                    camera_to_world_transform)

        if not array_of_camera_to_world_transforms:
            self.get_logger().error(
                f"No valid camera-to-world transforms for camera {camera_id}. Returning identity transform.")
            return np.eye(4)

        stacked_transform = np.hstack(array_of_camera_to_world_transforms)
        num_blocks = stacked_transform.shape[1] // 4
        stacked_identity = np.hstack([np.eye(4) for _ in range(num_blocks)])

        pseudoinverse_result = np.linalg.pinv(stacked_transform)
        reliable_camera_transform = np.dot(
            stacked_identity, pseudoinverse_result)
        return reliable_camera_transform

    def find_random_max_index(self, scores_table):
        max_value = float('-inf')
        max_indices = []

        # Loop over the table to find the max value and its indices
        for marker_id in range(self.largest_marker + 1):
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
            self.get_logger().error(f"The center marker has no links between cameras nor markers")
        return max_marker_id
