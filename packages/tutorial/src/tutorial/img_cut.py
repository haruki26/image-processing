from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray


@dataclass
class CutCoordinates:
    f: Final[int]
    t: Final[int]


HEIGHT_CUT: Final[CutCoordinates] = CutCoordinates(f=0, t=200)
WIDTH_CUT: Final[CutCoordinates] = CutCoordinates(f=0, t=200)


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo]:
    img_cut = img[HEIGHT_CUT.f : HEIGHT_CUT.t, WIDTH_CUT.f : WIDTH_CUT.t, :]
    return (ProcessedImageInfo("ktech_cut", img_cut),)


if __name__ == "__main__":
    main()
