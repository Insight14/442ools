"""
Team classification via jersey-color clustering.
---------------------------------------------------
Approach: crop the top half of each player's bounding box (torso/jersey
area -- avoids shorts, socks, and grass at the bottom of the box), then use
KMeans to separate "player pixels" from "background pixels" within that
crop, and take the player-pixel cluster's average color as the jersey color.

Once we have jersey colors for a batch of players, a second KMeans (k=2)
groups them into two teams by color similarity.

Known limitation: this assumes exactly two dominant outfield colors.
Goalkeepers (who wear a different kit) and referees (a third color) will
get force-classified into one of the two teams, since we're not building
a 3+ class model here. That's an acceptable rough edge for now -- it
doesn't break tracking stabilization, it just means a keeper might get
labeled with the wrong team color occasionally.
"""

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}          # team_id (1 or 2) -> BGR color
        self.kmeans = None             # fitted 2-cluster model over team colors
        self._bootstrap_colors = []    # buffer of jersey colors collected pre-fit

    @property
    def is_ready(self) -> bool:
        return self.kmeans is not None

    def _get_background_cluster(self, clustered_image: np.ndarray) -> int:
        """Corner pixels are almost always background (pitch), not player."""
        corners = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        return max(set(corners), key=corners.count)

    def get_jersey_color(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """Extract the dominant jersey color from a player's bounding box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        top_half = crop[: max(1, crop.shape[0] // 2), :]
        if top_half.shape[0] < 2 or top_half.shape[1] < 2:
            return None

        pixels = top_half.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 4:
            return None

        km = KMeans(n_clusters=2, init="k-means++", n_init=3, random_state=0)
        km.fit(pixels)
        clustered = km.labels_.reshape(top_half.shape[0], top_half.shape[1])

        bg_cluster = self._get_background_cluster(clustered)
        player_cluster = 1 - bg_cluster
        return km.cluster_centers_[player_cluster]

    def add_bootstrap_sample(self, frame: np.ndarray, bbox) -> None:
        """Collect a jersey-color sample during the warmup phase, before
        the two team clusters have been established."""
        color = self.get_jersey_color(frame, bbox)
        if color is not None:
            self._bootstrap_colors.append(color)

    def try_fit(self, min_samples: int = 15) -> bool:
        """Attempt to fit the 2-team clustering once enough samples are
        collected. Returns True once fitting succeeds."""
        if self.is_ready:
            return True
        if len(self._bootstrap_colors) < min_samples:
            return False

        km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        km.fit(np.array(self._bootstrap_colors))
        self.kmeans = km
        self.team_colors[1] = km.cluster_centers_[0]
        self.team_colors[2] = km.cluster_centers_[1]
        return True

    def predict_team(self, frame: np.ndarray, bbox) -> int | None:
        """Returns 1 or 2, or None if team clustering isn't ready yet or
        a jersey color couldn't be extracted."""
        if not self.is_ready:
            return None
        color = self.get_jersey_color(frame, bbox)
        if color is None:
            return None
        team_id = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        return team_id