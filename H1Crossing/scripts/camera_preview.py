"""Compact display copies only; never resize sensor inputs to perception."""
import cv2


def preview_size(width, height):
    scale = min(1.0, 320 / width, 240 / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def preview_frame(rgb, bbox, lines):
    height, width = rgb.shape[:2]
    size = preview_size(width, height)
    picture = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    picture = cv2.resize(picture, size, interpolation=cv2.INTER_AREA)
    if bbox is not None:
        x, y, w, h = bbox
        sx, sy = size[0] / width, size[1] / height
        start = (round(x * sx), round(y * sy))
        end = (round((x + w - 1) * sx), round((y + h - 1) * sy))
        cv2.rectangle(picture, start, end, (0, 0, 255), 1)
    # Draw after shrinking so labels retain readable screen-space size.
    for index, line in enumerate(lines):
        origin = (4, 18 + index * 18)
        cv2.putText(picture, line, origin, 0, .40, (0, 0, 0), 3)
        cv2.putText(picture, line, origin, 0, .40, (255, 255, 255), 1)
    return picture
