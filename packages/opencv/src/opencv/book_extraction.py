from typing import TYPE_CHECKING, Final

import cv2

from shared.pipeline import ProcessedImageInfo

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

AREA_THRESHOLD: Final = 1000


@pipe.process("books.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < AREA_THRESHOLD:
            continue
        cv2.drawContours(img, [contour], -1, (0, 255, 255), 2)

    return ProcessedImageInfo("extraction_result.jpg", img)


if __name__ == "__main__":
    main()
