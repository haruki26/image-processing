from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray


@dataclass
class MaskCoordinates:
    f: Final[int]
    t: Final[int]


HEIGHT_MASK: Final[MaskCoordinates] = MaskCoordinates(f=0, t=200)
WIDTH_MASK: Final[MaskCoordinates] = MaskCoordinates(f=0, t=200)


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo]:
    img_mask = img.copy()
    img_mask[HEIGHT_MASK.f : HEIGHT_MASK.t, WIDTH_MASK.f : WIDTH_MASK.t, :] = 0
    return (ProcessedImageInfo("ktech_mask", img_mask),)


if __name__ == "__main__":
    main()
