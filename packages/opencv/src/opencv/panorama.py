from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.pipeline import ProcessedImageInfo

from opencv.utils.image import image_manager
from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

RATIO: Final = 0.8


@pipe.process("picture1.png")
def main(img1: ImageArray) -> ProcessedImageInfo:
    img2 = image_manager.read_image("picture2.png")

    akaze = cv2.AKAZE.create()
    (kps1, des1), (kps2, des2) = [akaze.detectAndCompute(i, None) for i in [img1, img2]]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [[m] for m, n in matches if m.distance < RATIO * n.distance]

    src_pts = []
    dst_pts = []
    for m in good_matches:
        x1, y1 = kps1[m[0].queryIdx].pt
        x2, y2 = kps2[m[0].trainIdx].pt
        src_pts.append((x1, y1))
        dst_pts.append((x2, y2))

    from_pts, to_pts = np.float32(src_pts), np.float32(dst_pts)

    M, _ = cv2.findHomography(to_pts, from_pts, cv2.RANSAC, 5.0)

    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]

    # img2 を img1 の座標系へ射影
    panorama = cv2.warpPerspective(img2, M, (width, height))

    # 左側に img1 を配置
    panorama[: img1.shape[0], : img1.shape[1]] = img1
    return ProcessedImageInfo("panorama1.jpg", panorama)


if __name__ == "__main__":
    main()
