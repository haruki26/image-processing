from pathlib import Path
from typing import TYPE_CHECKING, Final

import cv2

from shared.color import RGBColor

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray
    from shared.pipeline import ProcessedImageInfo

CASCADE_PATH: Final[Path] = Path(__file__).parent / "cascade" / "haarcascade_frontalface_default.xml"

SCALE_FACTOR: Final[float] = 1.1
MIN_NEIGHBORS: Final[int] = 2

BOX_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)


@pipe.process("face.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), thickness=2)
    return pipe.create_processed_image_info("face_detection.jpg", img)


if __name__ == "__main__":
    main()
