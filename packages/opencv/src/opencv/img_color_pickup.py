from typing import TYPE_CHECKING, Final, TypedDict

import cv2

from shared.color import HSVColor, RGBColor

from opencv.utils.image import image_manager

if TYPE_CHECKING:
    from shared.image import ImageArray


class MouseCallbackParams(TypedDict):
    img: ImageArray


def display_rgb_color(color: RGBColor) -> None:
    print(f"RGB Color: (R:{color.red}, G:{color.green}, B:{color.blue})")  # noqa: T201


def display_hsv_color(color: HSVColor) -> None:
    print(f"HSV Color: (H:{color.hue}, S:{color.saturation}, V:{color.value})")  # noqa: T201


def display_color(
    rgb_color: RGBColor,
    hsv_color: HSVColor,
) -> None:
    display_rgb_color(rgb_color)
    display_hsv_color(hsv_color)


def handler(
    event: cv2.MouseEventTypes,
    x: int,
    y: int,
    _: cv2.MouseEventFlags,
    params: MouseCallbackParams,
) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = params["img"][y, x]
        rgb_color = RGBColor(red=int(bgr_color[2]), green=int(bgr_color[1]), blue=int(bgr_color[0]))

        hsv = cv2.cvtColor(params["img"], cv2.COLOR_BGR2HSV)
        hsv_color = HSVColor(
            hue=int(hsv[y, x][0]),
            saturation=int(hsv[y, x][1]),
            value=int(hsv[y, x][2]),
        )

        display_color(rgb_color, hsv_color)


ESC_KEY: Final[int] = 27


def main() -> None:
    img = image_manager.read_image("ball.jpg")

    cv2.namedWindow("Color Pickup")
    cv2.setMouseCallback(
        "Color Pickup",
        handler,  # ty:ignore[invalid-argument-type]
        {"img": img},
    )

    while True:
        cv2.imshow("Color Pickup", img)
        if cv2.waitKey(1) & 0xFF == ESC_KEY:  # ESC key
            break

    cv2.destroyAllWindows()

    image_manager.save_image(img, "ball_color_pickup.jpg")


if __name__ == "__main__":
    main()
