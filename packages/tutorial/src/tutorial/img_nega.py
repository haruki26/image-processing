from typing import TYPE_CHECKING

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo]:
    nega_img = 255 - img
    return (ProcessedImageInfo("ktech_negative", nega_img),)


if __name__ == "__main__":
    main()
