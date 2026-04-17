import cv2

from tutorial.utils import image_manager


def main() -> None:
    img = image_manager.read_image("ktech.jpg")

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
