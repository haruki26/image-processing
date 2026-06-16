from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.pipeline import ProcessedImageInfo

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

PT1: Final = (85, 52)
PT2: Final = (228, 115)
PT3: Final = (237, 332)
PT4: Final = (103, 422)
W: Final = 300
H: Final = 300
D: Final = 100
PP1: Final = (D, D)
PP2: Final = (W + D, D)
PP3: Final = (W + D, H + D)
PP4: Final = (D, H + D)
P_ORIGINAL = np.array([PT1, PT2, PT3, PT4], dtype=np.float32)
P_TRANS = np.array([PP1, PP2, PP3, PP4], dtype=np.float32)


@pipe.process("poster.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    trans = cv2.getPerspectiveTransform(P_ORIGINAL, P_TRANS)
    img = cv2.warpPerspective(img, trans, (W + 2 * D, H + 2 * D))

    return ProcessedImageInfo(
        image_name="poster_distorted.jpg",
        image_array=img,
    )


if __name__ == "__main__":
    main()
