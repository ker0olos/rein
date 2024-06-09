import cv2


def optimize_image(image, width=480, brightness=50):
    if image.shape[1] == width:
        return image

    aspect_ratio = float(image.shape[0]) / float(image.shape[1])

    height = round(width * aspect_ratio)

    resized_image = cv2.resize(image, (width, height))
    brightened_image = cv2.convertScaleAbs(resized_image, alpha=1, beta=brightness)

    return brightened_image
