from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Final

import cv2

from tutorial.utils.image import image_manager

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared import ImageArray


@dataclass
class ProcessedImageInfo:
    image_name: Final[str]
    image_array: Final[ImageArray]


def process_pipe(main_func: Callable[[ImageArray], tuple[ProcessedImageInfo, ...]]) -> Callable[[], None]:

    @wraps(main_func)
    def wrapper() -> None:
        img = image_manager.read_image("ktech.jpg")

        processed_images = main_func(img)
        for processed_image in processed_images:
            cv2.imshow(processed_image.image_name, processed_image.image_array)
            image_manager.save_image(
                processed_image.image_array, f"{processed_image.image_name.replace(' ', '_').lower()}.jpg"
            )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return wrapper
