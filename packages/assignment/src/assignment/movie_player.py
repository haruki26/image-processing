from pathlib import Path
from typing import Final

import cv2

from shared.color import RGBColor

VIDEO_PATH: Final = Path(__file__).parent.parent.parent / "movies" / "people_move.mp4"

ESC_KEY: Final = 27

WINDOW_WIDTH: Final = 800
WINDOW_HEIGHT: Final = 600

BUTTON_BACKGROUND: Final = RGBColor(255, 255, 0)
BUTTON_FOREGROUND: Final = RGBColor(0, 0, 0)
BUTTON_WIDTH: Final = 100
BUTTON_HEIGHT: Final = 50
BUTTON_GAP: Final = 20

PROGRESS_BAR_COLOR: Final = RGBColor(255, 255, 0)
PROGRESS_BAR_WIDTH: Final = WINDOW_WIDTH - 20
PROGRESS_BAR_HEIGHT: Final = 10
PROGRESS_BAR_BOTTOM_MARGIN: Final = 10
PROGRESS_BAR_LOC: Final = (
    10,
    WINDOW_HEIGHT - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_BOTTOM_MARGIN,
)

# プログレスバーとの間隔
BUTTON_PROGRESS_GAP: Final = 15

BUTTON_Y: Final = PROGRESS_BAR_LOC[1] - BUTTON_PROGRESS_GAP - BUTTON_HEIGHT

# 4個のボタン全体を中央寄せ
BUTTON_COUNT: Final = 4
BUTTONS_TOTAL_WIDTH: Final = BUTTON_WIDTH * BUTTON_COUNT + BUTTON_GAP * (BUTTON_COUNT - 1)
BUTTONS_START_X: Final = (WINDOW_WIDTH - BUTTONS_TOTAL_WIDTH) // 2

BACK_BUTTON_LOC: Final = (
    BUTTONS_START_X,
    BUTTON_Y,
)

PLAY_BUTTON_LOC: Final = (
    BUTTONS_START_X + BUTTON_WIDTH + BUTTON_GAP,
    BUTTON_Y,
)

STOP_BUTTON_LOC: Final = (
    BUTTONS_START_X + (BUTTON_WIDTH + BUTTON_GAP) * 2,
    BUTTON_Y,
)

FAST_FORWARD_BUTTON_LOC: Final = (
    BUTTONS_START_X + (BUTTON_WIDTH + BUTTON_GAP) * 3,
    BUTTON_Y,
)

BUTTON_LOCS: Final = {
    "Back": BACK_BUTTON_LOC,
    "Play": PLAY_BUTTON_LOC,
    "Stop": STOP_BUTTON_LOC,
    "FF": FAST_FORWARD_BUTTON_LOC,
}

FRAME_GOUNTER_COLOR: Final = RGBColor(255, 255, 0)

WINDOW_NAME: Final = "movie_player"

# FFボタン1回で進めるフレーム数
FF_SKIP_FRAMES: Final = 10


def calc_button_rect(loc: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = loc
    return (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT)


def is_inside_button(rect: tuple[tuple[int, int], tuple[int, int]], point: tuple[int, int]) -> bool:
    (x1, y1), (x2, y2) = rect
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def is_inside_progress_bar(point: tuple[int, int]) -> bool:
    x, y = point
    bar_x, bar_y = PROGRESS_BAR_LOC
    return bar_x <= x <= bar_x + PROGRESS_BAR_WIDTH and bar_y <= y <= bar_y + PROGRESS_BAR_HEIGHT


def button_handler(label: str, state: dict) -> None:
    if label == "Back":
        state["seek_frame"] = 0
        state["needs_frame"] = True
    elif label == "Play":
        state["is_paused"] = False
    elif label == "Stop":
        state["is_paused"] = True
    elif label == "FF":
        # 絶対位置へのシークは動画バックエンドによってEOF扱いになるため、
        # 再生ループ側でフレームを順に読み飛ばす。
        state["skip_frames"] += FF_SKIP_FRAMES
        state["needs_frame"] = True


def on_click(event: int, x: int, y: int, flags: int, state: dict) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if is_inside_progress_bar((x, y)):
        progress = (x - PROGRESS_BAR_LOC[0]) / PROGRESS_BAR_WIDTH
        state["seek_frame"] = min(
            int(progress * max(state["frame_count"] - 1, 0)),
            max(state["frame_count"] - 1, 0),
        )
        state["skip_frames"] = 0
        state["is_paused"] = False
        state["needs_frame"] = True
        return

    for label, rect in state["button_rects"].items():
        if is_inside_button(rect, (x, y)):
            button_handler(label, state)
            break


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        msg = f"動画を開けませんでした: {VIDEO_PATH}"
        raise RuntimeError(msg)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / frame_rate)) if frame_rate > 0 else 33
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    state = {
        "cap": cap,
        "button_rects": {label: calc_button_rect(loc) for label, loc in BUTTON_LOCS.items()},
        "frame_count": frame_count,
        "current_frame": 0,
        "is_paused": False,
        "needs_frame": True,
        "seek_frame": None,
        "skip_frames": 0,
    }
    cv2.setMouseCallback(WINDOW_NAME, on_click, state)

    displayed_frame = None

    try:
        while True:
            if state["seek_frame"] is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state["seek_frame"])
                state["current_frame"] = state["seek_frame"]
                state["seek_frame"] = None

            while state["skip_frames"] > 0:
                if not cap.grab():
                    state["is_paused"] = True
                    state["skip_frames"] = 0
                    break
                state["skip_frames"] -= 1

            if not state["is_paused"] or state["needs_frame"]:
                ret, frame = cap.read()
                if not ret:
                    state["is_paused"] = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                else:
                    state["current_frame"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    displayed_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                state["needs_frame"] = False

            if displayed_frame is None:
                continue

            screen = displayed_frame.copy()
            for label, rect in state["button_rects"].items():
                cv2.rectangle(
                    screen,
                    *rect,
                    BUTTON_BACKGROUND.get_tuple(),
                    -1,
                )
                cv2.putText(
                    screen,
                    label,
                    (rect[0][0] + 10, rect[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    BUTTON_FOREGROUND.get_tuple(),
                    2,
                )

            cv2.putText(
                screen,
                f"{state['current_frame']}/{frame_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                FRAME_GOUNTER_COLOR.get_tuple(),
                2,
            )

            progress = state["current_frame"] / max(frame_count, 1)
            progress_start = PROGRESS_BAR_LOC
            progress_end = (
                PROGRESS_BAR_LOC[0] + PROGRESS_BAR_WIDTH,
                PROGRESS_BAR_LOC[1] + PROGRESS_BAR_HEIGHT,
            )
            cv2.rectangle(
                screen,
                progress_start,
                progress_end,
                PROGRESS_BAR_COLOR.get_tuple(),
                1,
            )
            cv2.rectangle(
                screen,
                progress_start,
                (
                    PROGRESS_BAR_LOC[0] + int(progress * PROGRESS_BAR_WIDTH),
                    progress_end[1],
                ),
                PROGRESS_BAR_COLOR.get_tuple(),
                -1,
            )

            cv2.imshow(WINDOW_NAME, screen)

            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ESC_KEY):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
