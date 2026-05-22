from typing import Final

import cv2

from opencv.utils.image import image_manager
from opencv.utils.select_rois import select_rois

SELECT_WINDOW_NAME: Final[str] = "Select ROIS"

ESC_KEY: Final = 27


def main() -> None:
    img = image_manager.read_image("ktech.jpg")

    for roi in select_rois(img, window_name=SELECT_WINDOW_NAME):
        cv2.rectangle(img, roi, (0, 255, 255), 2)

    cv2.imshow("original", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
