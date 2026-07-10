from typing import TYPE_CHECKING, Final

import cv2

from shared.pipeline import ProcessedImageInfo

from opencv.utils.image import image_manager
from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shared import ImageArray

RATIO: Final = 0.8


AREA_THRESHOLD: Final = 1000


def detect_book_regions(img: ImageArray) -> list[NDArray]:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [c for c in contours if cv2.contourArea(c) > AREA_THRESHOLD]


@pipe.process("books.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    target_img = image_manager.read_image("book5.jpg")
    akaze = cv2.AKAZE.create()
    (_, des1), (kps2, des2) = [akaze.detectAndCompute(i, None) for i in [target_img, img]]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [[m] for m, n in matches if m.distance < RATIO * n.distance]

    contours = detect_book_regions(img)

    match_num = [0] * len(contours)
    for [match] in good_matches:
        pt = kps2[match.trainIdx].pt
        for i, contour in enumerate(contours):
            if cv2.pointPolygonTest(contour, pt, measureDist=False) >= 0:
                match_num[i] += 1

    max_match = max(match_num)
    if max_match == 0:
        return ProcessedImageInfo("no_book.jpg", img)

    max_index = match_num.index(max_match)
    contour = contours[max_index]
    x, y, w, h = cv2.boundingRect(contour)
    selected_img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 255), 2)
    return ProcessedImageInfo("selected_book.jpg", selected_img)


if __name__ == "__main__":
    main()
