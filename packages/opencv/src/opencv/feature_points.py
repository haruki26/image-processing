from typing import TYPE_CHECKING

import cv2

from shared.pipeline import ProcessedImageInfo

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray


@pipe.process("poster_2.png")
def main(img: ImageArray) -> ProcessedImageInfo:
    akaze = cv2.AKAZE.create()
    kps, _ = akaze.detectAndCompute(img, None)
    img_akaze = cv2.drawKeypoints(img, kps, img, flags=4)

    return ProcessedImageInfo(
        image_name="poster_2_akaze.png",
        image_array=img_akaze,
    )


if __name__ == "__main__":
    main()
