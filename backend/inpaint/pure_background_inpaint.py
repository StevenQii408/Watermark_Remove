"""Fast background reconstruction for subtitles over flat or slowly varying areas."""

import cv2
import numpy as np


class PureBackgroundInpaint:
    def __init__(self, variance_threshold=180, temporal_window=5):
        self.variance_threshold = variance_threshold
        self.temporal_window = max(1, temporal_window)

    @staticmethod
    def _ring(mask):
        kernel = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        return (dilated > 0) & (mask == 0)

    def is_pure_background(self, frames, mask):
        if not frames or not np.any(mask):
            return False
        binary = (mask > 0).astype(np.uint8)
        ring = self._ring(binary)
        if ring.sum() < 32:
            return False
        values = []
        temporal_reference = None
        sample_frames = frames[::max(1, len(frames) // self.temporal_window)]
        for frame in sample_frames:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            ring_lab = lab[ring]
            values.append(ring_lab.var(axis=0).mean())
            if temporal_reference is None:
                temporal_reference = np.median(ring_lab, axis=0)
            elif np.linalg.norm(np.median(ring_lab, axis=0) - temporal_reference) > max(8.0, self.variance_threshold * 0.08):
                return False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 120)
            if float(np.count_nonzero(edges[ring])) / max(1, int(ring.sum())) > 0.35:
                return False
        return float(np.mean(values)) <= self.variance_threshold

    @staticmethod
    def _fill_frame(frame, mask):
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return frame.copy()
        ring = PureBackgroundInpaint._ring(binary)
        ys, xs = np.where(ring)
        if len(xs) == 0:
            return frame.copy()
        # Median ring color is robust to compression noise and text outlines.
        color = np.median(frame[ring], axis=0).astype(np.float32)
        result = frame.astype(np.float32).copy()
        result[binary > 0] = color
        # Preserve a gentle gradient when the flat area is not perfectly uniform.
        if len(xs) >= 16:
            for channel in range(3):
                coeff = np.polyfit(xs.astype(np.float32), frame[ys, xs, channel].astype(np.float32), 1)
                x_grid = np.arange(frame.shape[1], dtype=np.float32)
                gradient = np.polyval(coeff, x_grid)
                result[:, :, channel][binary > 0] = gradient[np.where(binary > 0)[1]]
        # Feather only the boundary to avoid a visible hard rectangle.
        alpha = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 1.2)
        alpha = np.clip(alpha, 0, 1)[..., None]
        return np.clip(result * alpha + frame.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)

    def __call__(self, frames, mask):
        if not self.is_pure_background(frames, mask):
            return None
        return [self._fill_frame(frame, mask) for frame in frames]

    def force(self, frames, mask):
        return [self._fill_frame(frame, mask) for frame in frames]
