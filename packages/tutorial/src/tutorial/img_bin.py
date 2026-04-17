from typing import TYPE_CHECKING

import numpy as np

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray

IMAGE_BIN_THRESHOLD = 127


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo]:
    img_gray_lumi = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype(np.uint8)

    img_bin = (np.where(img_gray_lumi > IMAGE_BIN_THRESHOLD, 255, 0)).astype(np.uint8)
    return (ProcessedImageInfo("ktech_binary", img_bin),)


if __name__ == "__main__":
    main()
