import cairosvg
import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.svg.model import Model
from src.tracking.tracking import Tracking

app = FastAPI()


async def fake_video_streamer():
    vid = cv2.VideoCapture(0)

    try:
        with open("models/face.svg", "r") as model_file:
            tracking = Tracking()
            model = Model(model_file.read())
            tracking.set_model(model)
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                tracking.process(frame)
                buffer = cairosvg.svg2png(model.tostring().encode())
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/png\r\n\r\n" + bytearray(buffer) + b"\r\n"
                )
    finally:
        vid.release()


@app.get("/")
async def video_stream():
    return StreamingResponse(
        fake_video_streamer(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
