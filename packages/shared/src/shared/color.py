from dataclasses import dataclass
from typing import Final


@dataclass
class RGBColor:
    red: Final[int]
    green: Final[int]
    blue: Final[int]

    def get_tuple(self) -> tuple[int, int, int]:
        return self.blue, self.green, self.red


@dataclass
class HSVColor:
    hue: Final[int]
    saturation: Final[int]
    value: Final[int]

    def get_tuple(self) -> tuple[int, int, int]:
        return self.hue, self.saturation, self.value
