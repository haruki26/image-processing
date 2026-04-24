from typing import Final

import cv2

from opencv.utils.image import image_manager

KSIZE: Final[tuple[int, int]] = (5, 5)


def main() -> None:
    img = image_manager.read_image("ball.jpg")
    blurred_img = cv2.blur(img, KSIZE)

    cv2.imshow("Blurred Image", blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_manager.save_image(blurred_img, "ball_blurred.jpg")


if __name__ == "__main__":
    main()
