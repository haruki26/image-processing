from pathlib import Path
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.pipeline import ProcessedImageInfo

from assignment.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

CASCADE_PATH: Final = Path(__file__).parent / "cascade" / "haarcascade_frontalface_default.xml"

SCALE_FACTOR: Final = 1.2
MIN_NEIGHBORS: Final = 2
MIN_SIZE: Final = (30, 30)

MAX_BESIDE_IMGS: Final = 4


@pipe.process("faces.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE
    )
    face_imgs = [img[y : y + h, x : x + w] for (x, y, w, h) in faces]

    base_height, base_width = min([(h, w) for (_, _, w, h) in faces])
    max_beside_imgs = min(MAX_BESIDE_IMGS, len(face_imgs))

    face_imgs = [cv2.resize(fi, (base_width, base_height)) for fi in face_imgs]
    row_imgs = []
    for i in range(0, len(face_imgs), max_beside_imgs):
        row = None
        if len(face_imgs[i : i + max_beside_imgs]) != max_beside_imgs:
            pad_width = max_beside_imgs - len(face_imgs[i : i + max_beside_imgs])
            pad_imgs = [np.zeros((base_height, base_width, 3), dtype=np.uint8) for _ in range(pad_width)]
            row = cv2.hconcat([*face_imgs[i : i + max_beside_imgs], *pad_imgs])
        else:
            row = cv2.hconcat(face_imgs[i : i + max_beside_imgs])

        row_imgs.append(row)

    for row in row_imgs:
        print(row.shape)
    result = cv2.vconcat(row_imgs) if row_imgs else None

    if result is None:
        msg = "No faces detected"
        raise ValueError(msg)

    return ProcessedImageInfo(
        image_name="faces.jpg",
        image_array=result,
    )


if __name__ == "__main__":
    main()
