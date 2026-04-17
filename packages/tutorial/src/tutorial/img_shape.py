from tutorial.utils import image_manager


def main() -> None:
    img = image_manager.read_image("ktech.jpg")

    height, width, channels = img.shape
    print(f"Height: {height}, Width: {width}, Channels: {channels}")  # noqa: T201


if __name__ == "__main__":
    main()
