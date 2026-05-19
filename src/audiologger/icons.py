"""PIL-generated tray icons for idle / recording / transcribing."""
from PIL import Image, ImageDraw


SIZE = 64


def _make_icon(fg: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Filled circle, leaves a 4-px border
    draw.ellipse((4, 4, SIZE - 4, SIZE - 4), fill=fg + (255,))
    return img


def idle_icon() -> Image.Image:
    return _make_icon((120, 120, 120))  # grey


def recording_icon() -> Image.Image:
    return _make_icon((220, 30, 30))    # red


def transcribing_icon() -> Image.Image:
    return _make_icon((220, 180, 30))   # yellow
