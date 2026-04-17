from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    import numpy as np

type ImageArray = np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]]


class ImageManager:
    def __init__(self, image_dir_path: Path | str) -> None:
        self.image_dir_path = Path(image_dir_path)

    def read_image(self, image_name: str) -> ImageArray:
        image_path = self.image_dir_path / image_name
        img = cv2.imread(image_path)

        if img is None:
            msg = f"Image not found at path: {image_path}"
            raise FileNotFoundError(msg)

        return img

    def save_image(self, image: ImageArray, output_name: str) -> None:
        output_path = self.image_dir_path / output_name
        cv2.imwrite(output_path, image)
