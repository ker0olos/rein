import cv2
import numpy as np
import resvg_python

from tests.utils import overlay_webcam
from src.svg.model import Model
from src.tracking.tracking import Tracking

if __name__ == "__main__":
    with open("tests/models/face.svg", "r") as model_file:
        model = Model(model_file.read())

    tracking = Tracking()
    vid = cv2.VideoCapture(0)
    tracking.set_model(model)

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        tracking.process(frame)
        image_bytes = resvg_python.svg_to_png(model.tostring())
        image_decoded = cv2.imdecode(np.array(bytearray(image_bytes)), cv2.IMREAD_COLOR)

        final_image = overlay_webcam(frame, image_decoded)

        cv2.imshow("rein", final_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            vid.release()
            cv2.destroyAllWindows()
            break
