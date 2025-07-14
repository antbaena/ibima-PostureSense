import mediapipe as mp
import cv2

class PoseAnnotator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def annotate(self, image: cv2.Mat, landmarks) -> cv2.Mat:
        annotated = image.copy()
        if landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=2))
        return annotated
