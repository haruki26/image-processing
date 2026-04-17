from typing import TYPE_CHECKING

import numpy as np

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo, ...]:
    img_gray_mean = img.mean(axis=2).astype(np.uint8)
    img_gray_lumi = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype(np.uint8)
    return (
        ProcessedImageInfo("ktech_gray_mean", img_gray_mean),
        ProcessedImageInfo("ktech_gray_lumi", img_gray_lumi),
    )


if __name__ == "__main__":
    main()
