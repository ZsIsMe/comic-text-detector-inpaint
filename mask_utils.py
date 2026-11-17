"""Mask geometry shared by RF-DETR text processing."""

import cv2
import numpy as np


DEFAULT_MASK_DILATE = 4
MASK_UNIFICATION_METHODS = {'none', 'rectangle', 'hull'}


def unify_mask(mask: np.ndarray, method: str) -> np.ndarray:
    if method == 'none':
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    points = np.concatenate(contours, axis=0)
    unified = np.zeros_like(mask)
    if method == 'rectangle':
        x, y, width, height = cv2.boundingRect(points)
        cv2.rectangle(unified, (x, y), (x + width, y + height), 255, -1)
    else:
        hull = cv2.convexHull(points)
        cv2.drawContours(unified, [hull], 0, 255, -1)
    return unified
