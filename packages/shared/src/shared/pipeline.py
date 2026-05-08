from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Final, overload

import cv2

from shared import ImageArray, ImageManager

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass
class ProcessedImageInfo:
    image_name: str
    image_array: ImageArray


type MainFn = Callable[[ImageArray], ProcessedImageInfo | tuple[ProcessedImageInfo, ...]]


class Pipeline:
    image_manager: Final[ImageManager]
    image_name: Final[str | None]

    @overload
    def __init__(self, image_name: str | None = None, *, images_dir: Path | str) -> None: ...
    @overload
    def __init__(self, image_name: str | None = None, *, image_manager: ImageManager) -> None: ...
    def __init__(
        self,
        image_name: str | None = None,
        images_dir: Path | str | None = None,
        image_manager: ImageManager | None = None,
    ) -> None:
        if images_dir is not None and image_manager is not None:
            msg = "Cannot specify both images_dir and image_manager"
            raise ValueError(msg)

        if images_dir is not None:
            self.image_manager = ImageManager(images_dir)
        elif image_manager is not None:
            self.image_manager = image_manager
        else:
            msg = "Must specify either images_dir or image_manager"
            raise ValueError(msg)

        self.image_name = image_name

    def create_processed_image_info(self, image_name: str, image_array: ImageArray) -> ProcessedImageInfo:
        return ProcessedImageInfo(image_name, image_array)

    def process(
        self, image_name: str | None = None, *, is_show_image: bool = True, is_save_image: bool = True
    ) -> Callable[[MainFn], Callable[[], None]]:

        def decorator(
            main_func: MainFn,
        ) -> Callable[[], None]:

            @wraps(main_func)
            def wrapper() -> None:
                target_image_name = image_name or self.image_name

                if target_image_name is None:
                    msg = "No image name specified. Please provide an image name either in the constructor or the decorator."  # noqa: E501
                    raise ValueError(msg)

                img = self.image_manager.read_image(target_image_name)

                result = main_func(img)
                for processed_image in (result,) if isinstance(result, ProcessedImageInfo) else result:
                    if is_show_image:
                        cv2.imshow(processed_image.image_name, processed_image.image_array)

                    if is_save_image:
                        self.image_manager.save_image(
                            processed_image.image_array,
                            f"{processed_image.image_name.replace(' ', '_').lower()}",
                        )

                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return wrapper

        return decorator


if __name__ == "__main__":
    from pathlib import Path

    pipe = Pipeline(images_dir=Path(__file__).parent.parent.parent.parent / "tutorial" / "images")

    @pipe.process("ktech.jpg")
    def main(img: ImageArray) -> ProcessedImageInfo:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pipe.create_processed_image_info("ktech_gray.jpg", gray)

    main()
