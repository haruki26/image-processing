from typing import Final

import cv2
import numpy as np

from shared.color import RGBColor

from opencv.utils.image import image_manager

WIDTH: Final[int] = 1280
HEIGHT: Final[int] = 960
GRID_SIZE: Final[int] = 80

CIRCLE_RADIUS: Final[int] = 200


# Black
GRID_LINE_COLOR: Final[RGBColor] = RGBColor(0, 0, 0)

# White
CANVAS_BASE_COLOR: Final[RGBColor] = RGBColor(255, 255, 255)

# Yellow
CIRCLE_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)
# Blue
CIRCLE_LINE_COLOR: Final[RGBColor] = RGBColor(0, 0, 255)


def main() -> None:
    img = np.full(
        (HEIGHT, WIDTH, 3),
        CANVAS_BASE_COLOR.get_tuple(),
        np.uint8,
    )

    height, width = img.shape[:2]

    for y in range(0, height, GRID_SIZE):
        cv2.line(img, (0, y), (width, y), GRID_LINE_COLOR.get_tuple(), 1)

    for x in range(0, width, GRID_SIZE):
        cv2.line(img, (x, 0), (x, height), GRID_LINE_COLOR.get_tuple(), 1)

    cv2.circle(img, (width // 2, height // 2), CIRCLE_RADIUS, CIRCLE_COLOR.get_tuple(), -1)
    cv2.circle(img, (width // 2, height // 2), CIRCLE_RADIUS, CIRCLE_LINE_COLOR.get_tuple(), 2)

    cv2.imshow("Circle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_manager.save_image(img, "circle.png")


if __name__ == "__main__":
    main()
