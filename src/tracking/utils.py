import cv2
import numpy as np


def optimize_image(image):
    width = 480

    if image.shape[1] == width:
        return image

    aspect_ratio = float(image.shape[0]) / float(image.shape[1])

    height = round(width * aspect_ratio)

    image = cv2.resize(image, (width, height))

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate the histogram of the grayscale image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # determine if the image is likely to be underexposed
    low_light_threshold = 15

    total_pixels = np.sum(hist)

    dark_pixels_sum = np.sum(hist[:low_light_threshold]) / total_pixels

    # if image is low-light
    # use a brighten grayscale version
    if dark_pixels_sum > 0.2:
        image = cv2.equalizeHist(gray)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.convertScaleAbs(image, alpha=1, beta=50)

    return image
