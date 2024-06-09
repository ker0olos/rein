import cv2
import numpy as np
import resvg_python

from src.svg.model import Model
from src.tracking.tracking import Tracking

if __name__ == "__main__":
    tracking = Tracking()

    vid = cv2.VideoCapture(0)

    with open("models/face.svg", "r") as model_file:
        model = Model(model_file.read())
        tracking.set_model(model)
        while True:
            ret, frame = vid.read()

            if not ret:
                break

            tracking.process(frame)
            image_bytes = resvg_python.svg_to_png(model.tostring())
            image_decoded = cv2.imdecode(
                np.array(bytearray(image_bytes)), cv2.IMREAD_COLOR
            )

            cv2.imshow("rein", image_decoded)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                vid.release()
                cv2.destroyAllWindows()
                break
