"""Local image IO with Unicode paths and unchanged OpenCV pixel formats."""

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png'}


def find_all_imgs(img_dir, abs_path=False):
    return [str(path.absolute()) if abs_path else path.name
            for path in Path(img_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def imread(img_path, read_type=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), read_type)


def imwrite(img_path, image, ext='.png'):
    # Existing exports use PNG even when the input path has a JPEG suffix.
    path = Path(img_path).with_suffix(ext)
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise OSError(f'無法編碼圖片：{path}')
    encoded.tofile(path)
