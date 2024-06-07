import cairosvg
import cv2
import numpy as np

from src.svg.model import Model
from src.tracking.tracking import Tracking

with open("models/face.svg", "r") as file:
    img = cv2.imread("tests/test_eyes_mouth.jpeg")
    numpy_array = np.array(img)
    #
    xml_string = file.read()
    model = Model(xml_string)
    #
    tracking = Tracking()
    tracking.set_model(model)
    tracking.process(numpy_array)
    # print(model.tostring())
    cairosvg.svg2png(bytestring=model.tostring().encode(), write_to="output.png")
