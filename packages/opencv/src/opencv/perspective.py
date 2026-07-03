from typing import TYPE_CHECKING

import cv2
import numpy as np

from shared.pipeline import ProcessedImageInfo

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    import numpy.typing as npt

    from shared import ImageArray


def order_points(
    pts: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    center = np.mean(pts, axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0],
    )

    ordered = pts[np.argsort(angles)]

    # 左上を先頭に
    start = np.argmin(ordered.sum(axis=1))
    ordered = np.roll(ordered, -start, axis=0)

    # 時計回り(TL→TR→BR→BL)になるよう補正
    v1 = ordered[1] - ordered[0]
    v2 = ordered[2] - ordered[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    if cross < 0:
        ordered = np.array(
            [ordered[0], ordered[3], ordered[2], ordered[1]],
            dtype=np.float32,
        )

    return ordered


def get_original_points(
    img: ImageArray,
) -> npt.NDArray[np.float32]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=lambda contour: cv2.contourArea(contour, oriented=False),
        reverse=True,
    )

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, closed=True)

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return order_points(pts)

    msg = "Book contour not found"
    raise ValueError(msg)


def calc_destination_points(
    src_pts: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], tuple[int, int]]:
    tl, tr, br, bl = src_pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    return dst, (width, height)


@pipe.process("book4.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    src_pts = get_original_points(img)

    dst_pts, dsize = calc_destination_points(src_pts)

    matrix = cv2.getPerspectiveTransform(
        src_pts,
        dst_pts,
    )

    warped = cv2.warpPerspective(
        img,
        matrix,
        dsize,
    )

    return ProcessedImageInfo(
        image_name="book4_corrected.jpg",
        image_array=warped,
    )


if __name__ == "__main__":
    main()
