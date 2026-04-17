from typing import TYPE_CHECKING, Final

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray

REDUCTION_RATIO: Final[float] = 0.5


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo]:
    img_reduced = img[:: int(1 / REDUCTION_RATIO), :: int(1 / REDUCTION_RATIO), :]
    return (ProcessedImageInfo("ktech_reduced", img_reduced),)


if __name__ == "__main__":
    main()
