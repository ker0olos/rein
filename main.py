import cairosvg
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.svg.model import Model
from src.tracking.tracking import Tracking

app = FastAPI()


async def fake_video_streamer():
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
        #
        output_image = cairosvg.svg2png(bytestring=model.tostring().encode())
        #
        yield (
            b"--frame\r\n"
            b"Content-Type: image/png\r\n\r\n" + bytearray(output_image) + b"\r\n"
        )


@app.get("/")
async def video_stream():
    return StreamingResponse(
        fake_video_streamer(), media_type="multipart/x-mixed-replace;boundary=frame"
    )
