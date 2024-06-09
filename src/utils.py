import cv2
import numpy as np


def resize_image(image: np.ndarray, width=240):
    if image.shape[1] == width:
        return image

    aspect_ratio = float(image.shape[0]) / float(image.shape[1])

    height = round(width * aspect_ratio)

    resized_image = cv2.resize(image, (width, height))
    # brightened_image = cv2.convertScaleAbs(
    #     resized_image, alpha=1, beta=brightness
    # )
    # return brightened_image
    return resized_image


def overlay_webcam(webcam_image, output_image, x=5, y=5):
    webcam_image = resize_image(webcam_image)
    # if img.ndim == 3 and img.shape[2] == 4:
    #     img = img[..., :3]
    start_x = x
    start_y = y

    output_image[
        start_y : start_y + webcam_image.shape[0],
        start_x : start_x + webcam_image.shape[1],
    ] = webcam_image

    return output_image
