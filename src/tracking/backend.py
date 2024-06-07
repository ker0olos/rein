from collections import namedtuple

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
)


BackendResults = namedtuple("BackendResults", ["face_landmarks_normalized"])


class Backend:
    def __init__(self):
        self._face = FaceLandmarker.create_from_options(options)

    def process(self, image):
        # optimized_image = resize_image(image, 480)

        face_landmarks_normalized = None
        face_landmarks = np.empty((478, 3))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        face_landmarker_result = self._face.detect(mp_image)

        if face_landmarker_result:
            for i, pt in enumerate(face_landmarker_result.face_landmarks[0]):
                face_landmarks[i][0] = pt.x
                face_landmarks[i][1] = pt.y
                face_landmarks[i][2] = pt.z

            xs, ys, _ = face_landmarks.T

            bbox = np.min(xs), np.max(xs), np.min(ys), np.max(ys)

            face_landmarks_normalized = np.array(
                [
                    [
                        (pt[0] - bbox[0]) / (bbox[1] - bbox[0]),
                        (pt[1] - bbox[2]) / (bbox[3] - bbox[2]),
                    ]
                    for pt in face_landmarks
                ]
            )

            # if pose tracking is enabled
            # if features == "experimental":
            #     results = self._pose.process(optimized_image)

            #     if results.pose_landmarks:
            #         pose_landmarks = np.empty((33, 4))

            #         for i, pt in enumerate(results.pose_landmarks.landmark):
            #             pose_landmarks[i][0] = pt.x
            #             pose_landmarks[i][1] = pt.y
            #             pose_landmarks[i][2] = pt.z
            #             pose_landmarks[i][3] = pt.visibility

        return BackendResults(face_landmarks_normalized)
