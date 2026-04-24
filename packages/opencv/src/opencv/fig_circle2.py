from typing import Final

import cv2

from shared.color import RGBColor

from opencv.utils.image import image_manager

WIDTH: Final[int] = 1280
HEIGHT: Final[int] = 960
GRID_SIZE: Final[int] = 80

CIRCLE_RADIUS: Final[int] = 100

GRID_LINE_COLOR: Final[RGBColor] = RGBColor(0, 0, 0)


def main() -> None:
    img = image_manager.read_image("ktech.jpg")

    height, width = img.shape[:2]

    for y in range(0, height, GRID_SIZE):
        cv2.line(img, (0, y), (width, y), GRID_LINE_COLOR.get_tuple(), 1)

    for x in range(0, width, GRID_SIZE):
        cv2.line(img, (x, 0), (x, height), GRID_LINE_COLOR.get_tuple(), 1)

    mean_color = cv2.mean(img)[:3]

    cv2.circle(img, (width // 2, height // 2), CIRCLE_RADIUS, mean_color, -1)

    cv2.imshow("Circle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
