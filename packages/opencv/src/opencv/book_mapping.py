from typing import TYPE_CHECKING, Final

import cv2

from shared.pipeline import ProcessedImageInfo

from opencv.utils.image import image_manager
from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

RATIO: Final = 0.8


@pipe.process("books.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    target_img = image_manager.read_image("book5.jpg")

    akaze = cv2.AKAZE.create()
    (kps1, des1), (kps2, des2) = [akaze.detectAndCompute(i, None) for i in [target_img, img]]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [[m] for m, n in matches if m.distance < RATIO * n.distance]

    img_matches = cv2.drawMatchesKnn(target_img, kps1, img, kps2, good_matches, None)  # ty:ignore[no-matching-overload]

    return ProcessedImageInfo("matching_result.jpg", img_matches)


if __name__ == "__main__":
    main()
