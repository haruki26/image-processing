from typing import TYPE_CHECKING, Final

import cv2

from shared.color import RGBColor
from shared.pipeline import ProcessedImageInfo

from opencv.utils.image import image_manager
from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

TEMP_IMAGE_NAME: Final = "temp.jpg"

RECT_COLOR: Final = RGBColor(red=0, green=255, blue=0)


@pipe.process("face.jpg")
def get_temp(img: ImageArray) -> ProcessedImageInfo:
    roi = cv2.selectROI("Select ROI", img)
    return ProcessedImageInfo(
        image_name=TEMP_IMAGE_NAME, image_array=img[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    )


def check_create_temp() -> bool:
    flag = input("Create temp image? (Y/n): ")
    return flag.lower() == "n"


ALGORITHMS: Final = {
    "SSD": cv2.TM_SQDIFF,
    "SAD": cv2.TM_SQDIFF_NORMED,
    "MCC": cv2.TM_CCORR_NORMED,
    "ZNCC": cv2.TM_CCOEFF_NORMED,
}


def select_algorithm() -> int:
    print("Select algorithm:")
    for i, name in enumerate(ALGORITHMS.keys(), start=1):
        print(f"{i}: {name}")
    choice = input(f"Enter algorithm number(1-{len(ALGORITHMS)}): ")

    try:
        return ALGORITHMS[list(ALGORITHMS.keys())[int(choice) - 1]]
    except IndexError, ValueError:
        print(f"Invalid choice: {choice}")
        return select_algorithm()


def to_grayscale(img: ImageArray) -> ImageArray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@pipe.process("face.jpg")
def match_temp(img: ImageArray) -> ProcessedImageInfo:
    temp = to_grayscale(image_manager.read_image(TEMP_IMAGE_NAME))
    gray_img = to_grayscale(img)

    algorithm = select_algorithm()

    match = cv2.matchTemplate(gray_img, temp, algorithm)
    _, _, min_loc, max_loc = cv2.minMaxLoc(match)
    loc = min_loc if algorithm in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else max_loc

    img = cv2.rectangle(img, loc, (loc[0] + temp.shape[1], loc[1] + temp.shape[0]), RECT_COLOR.get_tuple(), 2)

    return ProcessedImageInfo(image_name="matched.jpg", image_array=img)


def main() -> None:
    has_temp = False
    try:
        image_manager.read_image(TEMP_IMAGE_NAME)
        has_temp = True
    except FileNotFoundError:
        pass

    if not has_temp:
        if check_create_temp():
            get_temp()
        else:
            print("No temp image found.")
            return
    elif check_create_temp():
        get_temp()

    match_temp()


if __name__ == "__main__":
    main()
