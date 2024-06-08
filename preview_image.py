import cairosvg
import cv2
import numpy as np

from src.svg.model import Model
from src.tracking.tracking import Tracking

if __name__ == "__main__":
    with open("models/face.svg", "r") as model_file:
        tracking = Tracking()
        model = Model(model_file.read())
        tracking.set_model(model)

        # frame = cv2.imread("tests/test_normal.jpeg")
        # frame = cv2.imread("tests/test_eyes_mouth.jpeg")
        frame = cv2.imread("tests/test_head_tilted_2.jpeg")
        tracking.process(frame)

        while tracking.last_frame_ms is None:
            pass

        image_bytes = cairosvg.svg2png(model.tostring().encode())
        image_decoded = cv2.imdecode(np.array(bytearray(image_bytes)), cv2.IMREAD_COLOR)

        while True:
            cv2.imshow("rein", image_decoded)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
