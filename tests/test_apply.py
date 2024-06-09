import os
from glob import glob

import cv2
import numpy as np
import pytest
import resvg_python

from src.svg.model import Model
from src.tracking.tracking import Tracking
from tests.utils import compare_img, overlay_webcam

with open("tests/models/face.svg") as file:
    face_model_file = file.read()

with open("tests/models/normal.svg") as file:
    normal_model_file = file.read()


@pytest.mark.parametrize("file", glob("tests/images/test_*"))
def test_basic_model(file):
    model = Model(face_model_file)

    tracking = Tracking()
    tracking.set_model(model)

    frame = cv2.imread(file)

    tracking.process(frame)

    while tracking.last_frame_ms is None:
        pass

    image_bytes = resvg_python.svg_to_png(model.tostring())
    image_decoded = cv2.imdecode(np.array(bytearray(image_bytes)), cv2.IMREAD_COLOR)

    final_image = overlay_webcam(frame, image_decoded)

    assert compare_img(os.path.basename(file), "basic_model", final_image) < 4.5


@pytest.mark.parametrize("file", glob("tests/images/test_*"))
def test_normal_model(file):
    model = Model(normal_model_file)

    tracking = Tracking()
    tracking.set_model(model)

    frame = cv2.imread(file)

    tracking.process(frame)

    while tracking.last_frame_ms is None:
        pass

    image_bytes = resvg_python.svg_to_png(model.tostring())
    image_decoded = cv2.imdecode(np.array(bytearray(image_bytes)), cv2.IMREAD_COLOR)

    final_image = overlay_webcam(frame, image_decoded)

    assert compare_img(os.path.basename(file), "normal_model", final_image) < 4.5
