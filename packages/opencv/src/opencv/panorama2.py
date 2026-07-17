from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.pipeline import ProcessedImageInfo

from opencv.utils.image import image_manager
from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

RATIO: Final = 0.8


def calc_homography(img1: ImageArray, img2: ImageArray) -> tuple[np.ndarray, list]:
    akaze = cv2.AKAZE.create()
    (kps1, des1), (kps2, des2) = [akaze.detectAndCompute(i, None) for i in (img1, img2)]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [[m] for m, n in matches if m.distance < RATIO * n.distance]

    src_pts = []
    dst_pts = []
    for m in good_matches:
        src_pts.append(kps1[m[0].queryIdx].pt)
        dst_pts.append(kps2[m[0].trainIdx].pt)

    # img2 -> img1
    M, _ = cv2.findHomography(
        np.float32(dst_pts),
        np.float32(src_pts),
        cv2.RANSAC,
        5.0,
    )

    return M, good_matches


def calc_panorama(base: ImageArray, images: list[tuple[ImageArray, np.ndarray]]) -> ImageArray:
    height = max([base.shape[0]] + [img.shape[0] for img, _ in images])
    width = base.shape[1] * (len(images) + 1)

    panorama = np.zeros((height, width, 3), dtype=base.dtype)
    panorama[: base.shape[0], : base.shape[1]] = base

    for img, M in images:
        warped = cv2.warpPerspective(img, M, (width, height))

        # 黒以外を重ねる
        mask = np.any(warped != 0, axis=2)
        panorama[mask] = warped[mask]

    return panorama


@pipe.process("hirosawa1.jpg")
def main(img1: ImageArray) -> ProcessedImageInfo:
    img2 = image_manager.read_image("hirosawa2.jpg")
    img3 = image_manager.read_image("hirosawa3.jpg")

    M21, _ = calc_homography(img1, img2)
    M32, _ = calc_homography(img2, img3)

    M31 = M21 @ M32

    panorama = calc_panorama(
        img1,
        [
            (img2, M21),
            (img3, M31),
        ],
    )

    return ProcessedImageInfo("panorama2.jpg", panorama)


if __name__ == "__main__":
    main()
