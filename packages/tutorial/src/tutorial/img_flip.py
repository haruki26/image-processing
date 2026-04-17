from typing import TYPE_CHECKING

from tutorial.utils import ProcessedImageInfo, process_pipe

if TYPE_CHECKING:
    from shared import ImageArray


@process_pipe
def main(img: ImageArray) -> tuple[ProcessedImageInfo, ...]:
    img_v_flip = img[::-1, :, :]
    img_h_flip = img[:, ::-1, :]
    return (
        ProcessedImageInfo("ktech_v_flip", img_v_flip),
        ProcessedImageInfo("ktech_h_flip", img_h_flip),
    )


if __name__ == "__main__":
    main()
