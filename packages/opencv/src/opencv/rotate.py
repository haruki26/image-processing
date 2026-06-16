from typing import Final

import cv2
import numpy as np

from opencv.utils.image import image_manager

WINDOW_PADDING: Final = 10
WINDOW_NAME: Final = "Rotated Image"

ROTATE_SPAN_MS: Final = 10
STOP_KEY: Final = "q"


def calc_window_size(w: int, h: int) -> int:
    return int((w**2 + h**2) ** 0.5) + WINDOW_PADDING


def calc_offset(window_size: int, h: int, w: int) -> tuple[int, int]:
    return (window_size - w) // 2, (window_size - h) // 2


def main() -> None:
    img = image_manager.read_image("ktech.jpg")
    h, w = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    window_size = calc_window_size(w, h)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    angle = 0
    while True:
        trans = cv2.getRotationMatrix2D(center, angle, 1.0)
        trans[0, 2] += window_size / 2 - center[0]
        trans[1, 2] += window_size / 2 - center[1]

        rotate_img = cv2.warpAffine(img, trans, (window_size, window_size))

        canvas = np.zeros((window_size, window_size, 3), dtype=np.uint8)
        h, w = rotate_img.shape[:2]
        x, y = calc_offset(window_size, h, w)
        canvas[y : y + h, x : x + w] = rotate_img

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(ROTATE_SPAN_MS)
        if key == ord(STOP_KEY):
            break
        angle += 1


if __name__ == "__main__":
    main()
