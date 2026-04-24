from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypedDict

import cv2

from shared.color import RGBColor

from opencv.utils.image import image_manager

if TYPE_CHECKING:
    from shared.image import ImageArray


@dataclass
class CircleInfo:
    radius: Final[int]
    color: Final[RGBColor]
    line_thickness: Final[int]


LEFT_CLICK_CIRCLE_INFO: Final[CircleInfo] = CircleInfo(radius=50, color=RGBColor(255, 0, 0), line_thickness=2)
RIGHT_CLICK_CIRCLE_INFO: Final[CircleInfo] = CircleInfo(radius=50, color=RGBColor(0, 0, 255), line_thickness=2)


class MouseCallbackParams(TypedDict):
    img: ImageArray


def draw_circle(
    event: cv2.MouseEventTypes,
    x: int,
    y: int,
    _: cv2.MouseEventFlags,
    params: MouseCallbackParams,
) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(
            params["img"],
            (x, y),
            LEFT_CLICK_CIRCLE_INFO.radius,
            LEFT_CLICK_CIRCLE_INFO.color.get_tuple(),
            LEFT_CLICK_CIRCLE_INFO.line_thickness,
        )
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(
            params["img"],
            (x, y),
            RIGHT_CLICK_CIRCLE_INFO.radius,
            RIGHT_CLICK_CIRCLE_INFO.color.get_tuple(),
            RIGHT_CLICK_CIRCLE_INFO.line_thickness,
        )


ESC_KEY: Final[int] = 27


def main() -> None:
    img = image_manager.read_image("ktech.jpg")

    cv2.namedWindow("Mouse Circle")
    cv2.setMouseCallback("Mouse Circle", draw_circle, {"img": img})  # ty:ignore[invalid-argument-type]

    while True:
        cv2.imshow("Mouse Circle", img)
        if cv2.waitKey(1) & 0xFF == ESC_KEY:  # ESC key to exit
            break
    cv2.destroyAllWindows()

    image_manager.save_image(img, "mouse_circle.jpg")


if __name__ == "__main__":
    main()
