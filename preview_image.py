import cv2
import numpy as np
import resvg_python

from src.svg.model import Model
from src.tracking.tracking import Tracking
from src.utils import overlay_webcam

if __name__ == "__main__":
    with open("models/face.svg", "r") as model_file:
        tracking = Tracking()
        model = Model(model_file.read())
        tracking.set_model(model)

        frame = cv2.imread("tests/test_normal.jpeg")
        frame = cv2.imread("tests/test_irises.jpeg")
        # frame = cv2.imread("tests/test_eyes_mouth.jpeg")
        # frame = cv2.imread("tests/test_head_tilted_2.jpeg")
        # frame = cv2.imread("tests/test_lowlight.jpeg")
        # frame = cv2.imread("tests/test_lowlight_2.jpeg")

        tracking.process(frame)

        while tracking.last_frame_ms is None:
            pass

        image_bytes = resvg_python.svg_to_png(model.tostring())
        image_decoded = cv2.imdecode(np.array(bytearray(image_bytes)), cv2.IMREAD_COLOR)

        final_image = overlay_webcam(frame, image_decoded)

        while True:
            cv2.imshow("rein", final_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
