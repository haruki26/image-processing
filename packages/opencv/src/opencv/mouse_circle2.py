from typing import TYPE_CHECKING, Final, TypedDict

import cv2

from shared.color import RGBColor

from opencv.utils.image import image_manager

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared.image import ImageArray

CIRCLE_RADIUS: Final[int] = 50
CIRCLE_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)
CIRCLE_LINE_THICKNESS: Final[int] = 2


class MouseCallbackParams(TypedDict):
    img: ImageArray
    set_prev_img: Callable[[], None]
    back_img: Callable[[], None]


def handler(
    event: cv2.MouseEventTypes,
    x: int,
    y: int,
    _: cv2.MouseEventFlags,
    params: MouseCallbackParams,
) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        params["set_prev_img"]()
        cv2.circle(
            params["img"],
            (x, y),
            CIRCLE_RADIUS,
            CIRCLE_COLOR.get_tuple(),
            CIRCLE_LINE_THICKNESS,
        )
    if event == cv2.EVENT_RBUTTONDOWN:
        params["back_img"]()


def main() -> None:
    img = image_manager.read_image("ktech.jpg")
    prev_imgs = [img.copy()]

    def set_prev_img() -> None:
        prev_imgs.append(img.copy())

    def back_img() -> None:
        if len(prev_imgs) == 0:
            return
        img[:] = prev_imgs.pop()

    cv2.namedWindow("Mouse Circle")
    cv2.setMouseCallback(
        "Mouse Circle",
        handler,  # ty:ignore[invalid-argument-type]
        {
            "img": img,
            "set_prev_img": set_prev_img,
            "back_img": back_img,
        },
    )

    while True:
        cv2.imshow("Mouse Circle", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
