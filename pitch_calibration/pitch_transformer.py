"""
Pixel -> real-world pitch coordinate transformer.
---------------------------------------------------
Loads a homography saved by calibrate.py and converts pixel positions
(e.g. a player's foot position in a video frame) into real-world pitch
coordinates in metres.

Includes a reliability check: a homography computed from a fixed reference
frame becomes inaccurate for pixel points far outside the region the
calibration points actually covered, or once the camera has panned/zoomed
away from the calibrated view. Rather than silently producing wrong-looking
metre values, is_reliable() flags when a point falls outside a safety
margin around the pitch bounds -- callers should skip displaying
speed/distance for those points rather than show a misleading number
(matches the "don't show misleading numbers when calibration is
unreliable" principle from the project README).
"""

import json

import cv2
import numpy as np

from pitch_calibration.pitch_reference import PITCH_LENGTH, PITCH_WIDTH


class PitchTransformer:
    def __init__(self, homography_path: str, out_of_bounds_margin_m: float = 5.0):
        with open(homography_path) as f:
            data = json.load(f)

        self.homography = np.array(data["homography"], dtype=np.float64)
        self.mean_reprojection_error_m = data.get("mean_reprojection_error_m")
        self.out_of_bounds_margin_m = out_of_bounds_margin_m

    def pixel_to_pitch(self, pixel_point) -> tuple:
        """pixel_point: (x, y) in image pixels. Returns (x, y) in metres,
        pitch-relative coordinates (0,0) to (PITCH_LENGTH, PITCH_WIDTH)."""
        pt = np.array([[pixel_point]], dtype=np.float64)  # shape (1,1,2)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        x, y = transformed[0, 0]
        return float(x), float(y)

    def is_reliable(self, pitch_point) -> bool:
        """A transformed point that lands far outside the pitch's actual
        bounds (beyond a safety margin) indicates the homography is being
        extrapolated past where it was calibrated -- not trustworthy for
        speed/distance display."""
        x, y = pitch_point
        m = self.out_of_bounds_margin_m
        return (-m <= x <= PITCH_LENGTH + m) and (-m <= y <= PITCH_WIDTH + m)

    def pixel_to_pitch_checked(self, pixel_point):
        """Convenience wrapper: returns (pitch_point, is_reliable). Callers
        that compute speed/distance should skip the frame (not display a
        value) when is_reliable is False, rather than showing a number
        derived from an untrustworthy transform."""
        pitch_point = self.pixel_to_pitch(pixel_point)
        return pitch_point, self.is_reliable(pitch_point)