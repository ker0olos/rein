import time

import numpy as np

from src.svg.model import Model
from src.tracking.backend import Backend
from src.tracking.filters import OneEuroFilter


def _origin(bbox, px, py):
    xmin, xmax, ymin, ymax = bbox
    width, height = xmax - xmin, ymax - ymin
    cx = xmin + (width / 2)
    cy = ymin + (height / 2)
    return (
        cx + (px * width),
        cy + (py * height),
    )


class Tracking(Backend):
    def __init__(self):
        self._backend = super()
        self._backend.__init__()

        self.__filter = {}

        self._face_landmarks = None
        # self._pose_landmarks = None

        self._model = None

    def set_model(self, model: Model):
        self._model = model

    def _getn_face(self, indices):
        return np.array([self._face_landmarks[i] for i in indices])

    # def _get_pose(self, indices):
    #     return np.array([self._pose_landmarks[i] for i in indices])

    def __filter__(self, key, n):
        if n is not None:
            t = time.time()

            # key has an active filter
            # process the new value
            # and return it
            if key in self.__filter:
                return self.__filter[key](t, n)
            # create a new filter
            # and return the same initialized value
            else:
                self.__filter[key] = OneEuroFilter(t, n)

        return n

    def process(self, image: np.ndarray):
        results = super().process(image)

        if results.face_landmarks_normalized is not None:
            self._face_landmarks = results.face_landmarks_normalized

        # if results.pose_landmarks is not None:
        #     self._pose_landmarks = results.pose_landmarks

        if self._model is not None:
            # revert model to its original state

            for p in self._model.paths:
                p._modified_segments = p._segments

            # apply the tracking to model
            if self._face_landmarks is not None:
                self.__apply_face_to_model__()

            # if self._pose_landmarks is not None:
            #     self.apply_pose_to_model()

    def __apply_face_to_model__(self):
        x, y = 0, 0.35

        if self._model.__exists__("face"):
            x = float(self._model.__find_attrs_by_id__("face", "rein:xpivot", x))
            y = float(self._model.__find_attrs_by_id__("face", "rein:ypivot", y))

            self._model.face_rotation = self.__get_face_tilt__()

            self._model.face_origin = _origin(self._model.face_virgin_bbox, x, y)

            self._model.__apply_eyes__(
                self.__get_eye_height__(),
                self.__get_eyebrow_diff__(),
                self.__get_iris_diff__(),
            )

            self._model.__apply_mouth__(*self.__get_mouth_size__())

    def __get_eyebrow_diff__(self):
        def _get(indices):
            a, b = self._getn_face(indices)
            return np.sqrt(np.sum((a - b) ** 2))

        return (
            self.__filter__("eyebrow-0", _get([105, 160])),
            self.__filter__("eyebrow-1", _get([334, 387])),
        )

    def __get_eye_height__(self):
        def _get(indices):
            a, b = self._getn_face(indices)
            return np.sqrt(np.sum((a - b) ** 2))

        return (
            self.__filter__("eye-0", _get([145, 159])),
            self.__filter__("eye-1", _get([374, 386])),
        )

    def __get_mouth_size__(self):
        x0, x1 = self._getn_face([61, 291])
        y0, y1 = self._getn_face([0, 17])

        w = x1[0] - x0[0]
        h = y1[1] - y0[1]

        return self.__filter__("mouth-0", w), self.__filter__("mouth-1", h)

    def __get_iris_diff__(self):
        LEFT_EYE = list([33, 133, 159, 145])
        RIGHT_EYE = list([362, 263, 386, 374])

        LEFT_IRIS = list([469, 471, 470, 472])
        RIGHT_IRIS = list([474, 476, 475, 477])

        def _get(EYE, IRIS):
            eye_a = self._getn_face(EYE[:2])
            eye_b = self._getn_face(EYE[2:4])

            iris_a = self._getn_face(IRIS[:2])
            iris_b = self._getn_face(IRIS[2:4])

            eye_center = np.array(
                [((eye_a[0] + eye_a[1]) / 2)[0], ((eye_b[0] + eye_b[1]) / 2)[1]]
            )
            iris_center = np.array(
                [((iris_a[0] + iris_a[1]) / 2)[0], ((iris_b[0] + iris_b[1]) / 2)[1]]
            )

            return iris_center - eye_center

        left, right = _get(LEFT_EYE, LEFT_IRIS), _get(RIGHT_EYE, RIGHT_IRIS)

        return self.__filter__("irises", (left + right) / 2)

    def __get_face_tilt__(self):
        a, b = self._getn_face([123, 352])

        cx, cy = (a + b) / 2

        n = 360 - (
            90
            - round(
                np.degrees(
                    np.arccos(
                        (a[1] - cy) / np.sqrt((a[0] - cx) ** 2 + (a[1] - cy) ** 2)
                    )
                ),
                6,
            )
        )

        return self.__filter__("face-0", n)

    # def get_arms_tilt(self):
    #     min_visibility = self._model.sensitivity

    #     def _get(a, b, counterclockwise):
    #         a, b = self._get_pose([a, b])

    #         if min_visibility > b[3]:
    #             return None

    #         cx, cy, _, _ = (a + b) / 2

    #         angle = round(
    #             180
    #             - np.degrees(
    #                 np.arccos(
    #                     (a[1] - cy) / np.sqrt((a[0] - cx) ** 2 + (a[1] - cy) ** 2)
    #                 )
    #             ),
    #             6,
    #         )

    #         return 360 - angle if counterclockwise else angle

    #     return (
    #         self.filter("arm-0", _get(12, 14, False)),
    #         self.filter("arm-1", _get(11, 13, True)),
    #     )

    #     # return _get(12, 14), _get(14, 16), 360 - _get(11, 13), 360 - _get(13, 15)

    # def get_face_turn(self):
    #     a, b = self._get([123, 352])
    #     cx, cy, cz = (a + b) / 2
    #     return round(acos((a[2] - cz) / sqrt((a[0] - cx) ** 2 + (a[2] - cz) ** 2)), 3)

    # def get_face_depth(self):
    #     a, b = self._get([9, 200])
    #     cx, cy, cz = (a + b) / 2
    #     return round(acos((a[2] - cz) / sqrt((a[1] - cy) ** 2 + (a[2] - cz) ** 2)), 3)
