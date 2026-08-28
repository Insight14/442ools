"""
Track ID stabilizer.
---------------------------------------------------
ByteTrack assigns its own internal tracker IDs based purely on box overlap
between consecutive frames. When that geometric signal breaks (players
clustering, brief occlusion, fast motion), ByteTrack drops the track and
starts a new ID -- which is the churn we measured in testing.

This layer sits on top of ByteTrack's raw output and re-stitches IDs using
an independent signal: when a brand-new raw ID appears close to where a
same-team player was last seen a moment ago, it's very likely a continuation
of that player, not a genuinely new one. We remap it to the earlier
"display ID" instead of showing a new number.

This is a heuristic, not a learned re-ID model -- it will not be perfect,
especially when two same-team players cross paths near each other. But it
should measurably reduce ID churn versus raw ByteTrack output, without
needing a trained appearance model.
"""

from dataclasses import dataclass, field
import math


@dataclass
class _TrackMemory:
    display_id: int
    team: int
    last_center: tuple
    last_frame: int


class TrackStabilizer:
    def __init__(self, match_distance_px: float = 80.0, max_frame_gap: int = 45):
        self.match_distance_px = match_distance_px
        self.max_frame_gap = max_frame_gap

        self.raw_to_display: dict[int, int] = {}   # ByteTrack raw id -> our stable display id
        self.memory: dict[int, _TrackMemory] = {}   # display_id -> last known state
        self._next_display_id = 1

    def _new_display_id(self) -> int:
        did = self._next_display_id
        self._next_display_id += 1
        return did

    def _find_match(self, team: int, center: tuple, frame_idx: int):
        """Find a recently-lost, same-team display track near this position."""
        best_id, best_dist = None, None
        for display_id, mem in self.memory.items():
            if mem.team != team:
                continue
            gap = frame_idx - mem.last_frame
            if gap <= 0 or gap > self.max_frame_gap:
                continue
            dist = math.dist(mem.last_center, center)
            if dist > self.match_distance_px:
                continue
            if best_dist is None or dist < best_dist:
                best_id, best_dist = display_id, dist
        return best_id

    def update(self, raw_id: int, team: int | None, center: tuple, frame_idx: int) -> int:
        """
        Returns a stable display_id for this detection.
        `team` should be None for non-person detections (e.g. the ball) --
        those pass through without team-based stitching.
        """
        if team is None:
            # No team signal available -- just pass the raw id through under
            # its own namespace so it doesn't collide with person display ids.
            return raw_id + 1_000_000  # simple namespace offset for non-person tracks

        if raw_id in self.raw_to_display:
            display_id = self.raw_to_display[raw_id]
        else:
            matched = self._find_match(team, center, frame_idx)
            display_id = matched if matched is not None else self._new_display_id()
            self.raw_to_display[raw_id] = display_id

        self.memory[display_id] = _TrackMemory(display_id, team, center, frame_idx)
        return display_id