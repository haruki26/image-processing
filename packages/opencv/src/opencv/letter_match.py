from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.color import RGBColor
from shared.pipeline import ProcessedImageInfo

from opencv.utils.image import image_manager
from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

CORR_IMAGE_NAME: Final = "correct.jpg"
TARGET_IMAGE: Final = "description-image.jpg"

FILL_COLOR: Final = RGBColor(red=255, green=255, blue=255)
RECT_COLOR: Final = RGBColor(red=255, green=0, blue=0)

THRESHOLD: Final = 0.9


@pipe.process(TARGET_IMAGE)
def get_letter(img: ImageArray) -> ProcessedImageInfo:
    roi = cv2.selectROI("Select ROI", img)
    return ProcessedImageInfo(
        image_name=CORR_IMAGE_NAME, image_array=img[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    )


def to_grayscale(img: ImageArray) -> ImageArray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@pipe.process(TARGET_IMAGE)
def match_incorrect_letter(img: ImageArray) -> ProcessedImageInfo:
    temp = to_grayscale(image_manager.read_image(CORR_IMAGE_NAME))
    gray_img = to_grayscale(img)

    match = cv2.matchTemplate(gray_img, temp, cv2.TM_CCOEFF_NORMED)
    loc = np.where(match >= THRESHOLD)
    fill_correct = img.copy()
    for pt in zip(*loc[::-1], strict=True):
        cv2.rectangle(fill_correct, pt, (pt[0] + temp.shape[1], pt[1] + temp.shape[0]), FILL_COLOR.get_tuple(), -1)

    reverse_color = cv2.bitwise_not(fill_correct)
    blur = cv2.blur(reverse_color, (5, 5))
    reverse_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(reverse_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = bin_img.shape
    img_area = img_h * img_w
    temp_area = temp.shape[0] * temp.shape[1]

    min_contour_area = max(20.0, temp_area * 0.1)
    max_bbox_area = img_area * 0.5

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        contour_area = cv2.contourArea(contour)

        if bbox_area >= max_bbox_area:
            continue

        if contour_area < min_contour_area:
            continue

        img = cv2.rectangle(img, (x, y), (x + w, y + h), RECT_COLOR.get_tuple(), 2)

    return ProcessedImageInfo(image_name=f"matched_{TARGET_IMAGE}", image_array=img)


def main() -> None:
    get_letter()
    match_incorrect_letter()


if __name__ == "__main__":
    main()
