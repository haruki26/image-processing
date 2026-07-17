from pathlib import Path
from typing import Final

import cv2

from opencv.utils.image import image_manager

VIDEO_PATH: Final = Path(__file__).parent.parent.parent / "movies" / "IMG_9041.MOV"

CAPTURE_FRAME: Final = 50


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    imgs = []

    step = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if step % CAPTURE_FRAME == 0:
            imgs.append(frame)
        step += 1

    stitcher = cv2.Stitcher.create()
    status, result = stitcher.stitch(imgs)

    if status == cv2.Stitcher_OK:
        cv2.imshow("Stitch Result", result)
        cv2.waitKey(0)
        image_manager.save_image(result, "stitch_result.jpg")
    else:
        print("Stitching failed code:", status)


if __name__ == "__main__":
    main()
