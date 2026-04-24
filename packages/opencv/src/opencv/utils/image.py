from pathlib import Path
from typing import Final

from shared import ImageManager

IMAGE_DIR: Final[Path] = Path(__file__).parent.parent.parent.parent / "images"

image_manager = ImageManager(IMAGE_DIR)
