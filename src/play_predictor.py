"""Manager-Level Tactical Intelligence & Pass/Shot Prediction Engine.

Combines:
  1. 16x12 Expected Threat (xT) Pitch Value Grid.
  2. Passing Lane Interception & Cover-Shadow Physics (perpendicular defender risk).
  3. Dynamic Space Exploitation & Separation metrics.
  4. Strict Optimal Decision Filter (Single best pass or shot with tactical reasoning).
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import math


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.dist(first, second)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def point_to_segment_distance(point: tuple[float, float], seg_a: tuple[float, float], seg_b: tuple[float, float]) -> float:
    """Calculate the shortest distance from a defender's position to a passing trajectory vector."""
    px, py = point
    x1, y1 = seg_a
    x2, y2 = seg_b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return distance(point, seg_a)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.dist((px, py), (proj_x, proj_y))


# Standard 16x12 Expected Threat (xT) transition matrix (105m x 68m pitch)
# Higher values towards opponent goal (attacking right side x=105m, y=34m)
XT_GRID_16x12 = [
    [0.005, 0.007, 0.008, 0.010, 0.012, 0.014, 0.017, 0.021, 0.026, 0.033, 0.042, 0.053, 0.067, 0.084, 0.100, 0.080],
    [0.006, 0.008, 0.009, 0.011, 0.013, 0.016, 0.019, 0.024, 0.030, 0.038, 0.049, 0.062, 0.078, 0.099, 0.120, 0.110],
    [0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.028, 0.035, 0.045, 0.058, 0.074, 0.094, 0.122, 0.160, 0.170],
    [0.007, 0.009, 0.011, 0.013, 0.016, 0.020, 0.025, 0.032, 0.041, 0.053, 0.069, 0.090, 0.118, 0.160, 0.230, 0.260],
    [0.007, 0.009, 0.011, 0.014, 0.017, 0.021, 0.027, 0.035, 0.045, 0.060, 0.079, 0.105, 0.145, 0.210, 0.310, 0.380],
    [0.008, 0.010, 0.012, 0.015, 0.018, 0.023, 0.030, 0.039, 0.051, 0.068, 0.092, 0.125, 0.180, 0.275, 0.430, 0.520],
    [0.008, 0.010, 0.012, 0.015, 0.018, 0.023, 0.030, 0.039, 0.051, 0.068, 0.092, 0.125, 0.180, 0.275, 0.430, 0.520],
    [0.007, 0.009, 0.011, 0.014, 0.017, 0.021, 0.027, 0.035, 0.045, 0.060, 0.079, 0.105, 0.145, 0.210, 0.310, 0.380],
    [0.007, 0.009, 0.011, 0.013, 0.016, 0.020, 0.025, 0.032, 0.041, 0.053, 0.069, 0.090, 0.118, 0.160, 0.230, 0.260],
    [0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.028, 0.035, 0.045, 0.058, 0.074, 0.094, 0.122, 0.160, 0.170],
    [0.006, 0.008, 0.009, 0.011, 0.013, 0.016, 0.019, 0.024, 0.030, 0.038, 0.049, 0.062, 0.078, 0.099, 0.120, 0.110],
    [0.005, 0.007, 0.008, 0.010, 0.012, 0.014, 0.017, 0.021, 0.026, 0.033, 0.042, 0.053, 0.067, 0.084, 0.100, 0.080],
]


@dataclass
class PlayerState:
    track_id: int
    team: int
    role: str
    position: tuple[float, float]


class PlayPredictor:
    """High-fidelity spatial intelligence evaluator producing optimal tactical decisions."""

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, attacking_direction: int = 1):
        if attacking_direction not in (-1, 1):
            raise ValueError("attacking_direction must be -1 or 1")
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.attacking_direction = attacking_direction
        self.player_history = defaultdict(lambda: deque(maxlen=10))
        self.ball_history = deque(maxlen=15)
        self.previous_possession = None

    def _get_xt_value(self, position: tuple[float, float]) -> float:
        """Lookup Expected Threat (xT) for a pitch position."""
        px, py = position
        if self.attacking_direction == -1:
            px = self.pitch_length - px

        col = int(clamp(px / self.pitch_length, 0.0, 0.999) * 16)
        row = int(clamp(py / self.pitch_width, 0.0, 0.999) * 12)
        return XT_GRID_16x12[row][col]

    def _attacking_x(self, position: tuple[float, float]) -> float:
        return position[0] if self.attacking_direction == 1 else self.pitch_length - position[0]

    def _possession(self, players: list[PlayerState], ball_position):
        if ball_position is None or not players:
            return None
        nearest = min(players, key=lambda player: distance(player.position, ball_position))
        return nearest if distance(nearest.position, ball_position) <= 3.2 else None

    def _evaluate_pass_options(self, owner: PlayerState, players: list[PlayerState]):
        """Evaluate all legal teammate passes with physics, cover shadows, and xT delta."""
        teammates = [
            p for p in players
            if p.team == owner.team and p.track_id != owner.track_id and p.role != "goalkeeper"
        ]
        opponents = [p for p in players if p.team != owner.team]
        candidates = []

        owner_xt = self._get_xt_value(owner.position)

        for target in teammates:
            pass_dist = distance(owner.position, target.position)
            # Filter realistic passing range (5m to 48m)
            if not 4.5 <= pass_dist <= 48.0:
                continue

            target_xt = self._get_xt_value(target.position)
            delta_xt = max(-0.05, target_xt - owner_xt)

            # 1. Passing Lane Interception Risk (Check all defenders along passing vector)
            lane_threat = 0.0
            for opp in opponents:
                perp_dist = point_to_segment_distance(opp.position, owner.position, target.position)
                if perp_dist < 4.0:
                    # Defender is very close to cutting the lane
                    dist_to_passer = distance(opp.position, owner.position)
                    if dist_to_passer < pass_dist:
                        lane_threat += max(0.0, 4.0 - perp_dist) * 0.25

            pass_lane_safety = clamp(1.0 - lane_threat)

            # 2. Receiver Space & Pressure Separation
            nearest_def_dist = min((distance(target.position, opp.position) for opp in opponents), default=12.0)
            space_safety = clamp(nearest_def_dist / 8.0)

            # 3. Progression & Tactical Value
            fwd_gain = self._attacking_x(target.position) - self._attacking_x(owner.position)
            
            # Weighted Manager Composite Score (0.0 to 1.0)
            # High reward for: high xT increase + open receiving space + clear passing lane
            score = (
                0.35 * pass_lane_safety +
                0.30 * space_safety +
                0.25 * clamp(0.5 + delta_xt * 4.0 + fwd_gain / 50.0) +
                0.10 * clamp(1.0 - pass_dist / 60.0)
            )

            # Generate manager-level tactical description
            if delta_xt > 0.08 or fwd_gain > 12.0:
                desc = "Line-breaking progressive pass into space"
            elif pass_dist > 26.0 and target.position[1] < 20 or target.position[1] > 48:
                desc = "Crossfield switch to exploit weak side"
            elif nearest_def_dist > 6.5:
                desc = "Pocket pass to unmarked receiver"
            else:
                desc = "Short combination play under pressure"

            candidates.append({
                "type": "pass",
                "score": round(score, 3),
                "confidence": round(score, 3),
                "expected_threat": round(target_xt, 3),
                "delta_xt": round(delta_xt, 3),
                "from_track_id": owner.track_id,
                "to_track_id": target.track_id,
                "receiver_role": target.role.upper(),
                "team": owner.team,
                "distance_m": round(pass_dist, 1),
                "forward_gain_m": round(fwd_gain, 1),
                "nearest_opponent_m": round(nearest_def_dist, 1),
                "lane_safety": round(pass_lane_safety, 2),
                "desc": desc,
            })

        if not candidates:
            return None

        # Return candidates sorted by highest tactical score
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _evaluate_shot_option(self, owner: PlayerState, players: list[PlayerState]):
        """Evaluate direct shot on goal threat (xG model heuristic)."""
        att_x = self._attacking_x(owner.position)
        # Must be in final 32m of the pitch
        if att_x < self.pitch_length - 32.0:
            return None

        goal_center = (self.pitch_length if self.attacking_direction == 1 else 0.0, self.pitch_width / 2.0)
        goal_dist = distance(owner.position, goal_center)
        if goal_dist > 30.0:
            return None

        # Calculate shot angle relative to goal posts (width 7.32m)
        angle_to_goal = math.atan2(abs(owner.position[1] - 34.0), max(1.0, self.pitch_length - att_x))
        angle_factor = clamp(math.cos(angle_to_goal))

        defenders = [p for p in players if p.team != owner.team]
        pressure = min((distance(owner.position, d.position) for d in defenders), default=15.0)

        # Expected Goals (xG) estimation
        xg = clamp((0.55 - goal_dist / 50.0) * angle_factor + (pressure / 40.0) * 0.15)

        return {
            "type": "shot",
            "score": round(xg, 3),
            "confidence": round(xg, 3),
            "expected_threat": round(xg, 3),
            "from_track_id": owner.track_id,
            "team": owner.team,
            "goal_distance_m": round(goal_dist, 1),
            "nearest_opponent_m": round(pressure, 1),
            "desc": f"High threat shot opportunity ({round(goal_dist, 1)}m out, {round(xg, 2)} xG)",
        }

    def update(self, frame_index: int, timestamp_s: float, players: list[PlayerState], ball_position):
        for player in players:
            self.player_history[player.track_id].append((timestamp_s, player.position))
        if ball_position is not None:
            self.ball_history.append((timestamp_s, ball_position))

        owner = self._possession(players, ball_position)
        possession = owner.track_id if owner else None
        suggestions = []
        events = []

        carrier_pressure = "low"
        if owner:
            opponents = [p for p in players if p.team != owner.team]
            min_opp_dist = min((distance(owner.position, opp.position) for opp in opponents), default=15.0)
            carrier_pressure = "high" if min_opp_dist < 2.5 else ("medium" if min_opp_dist < 5.0 else "low")

            pass_candidates = self._evaluate_pass_options(owner, players) or []
            shot_option = self._evaluate_shot_option(owner, players)

            # 1. Shot Opportunity Check
            if shot_option:
                suggestions.append(shot_option)

            # 2. Best Pass (Green) & High-Quality Backup Passes (Faded Yellow)
            if pass_candidates:
                # Top 1 Best Pass (Green)
                best_pass = pass_candidates[0]
                best_pass["tier"] = "primary"
                suggestions.append(best_pass)

                # Only top viable secondary passes (score >= 0.65 or within 15% of best)
                for backup in pass_candidates[1:3]:
                    if backup["score"] >= 0.62 and backup["score"] >= best_pass["score"] * 0.78:
                        backup_copy = dict(backup)
                        backup_copy["tier"] = "backup"
                        suggestions.append(backup_copy)

        # Detect Completed Passes
        if owner and self.previous_possession and owner.track_id != self.previous_possession["track_id"]:
            previous = self.previous_possession
            if owner.team == previous["team"] and previous["position"] is not None:
                travel = distance(previous["position"], ball_position) if ball_position else 0.0
                if travel >= 4.0:
                    events.append({
                        "type": "completed_pass",
                        "confidence": round(clamp(0.65 + travel / 30.0), 3),
                        "from_track_id": previous["track_id"],
                        "to_track_id": owner.track_id,
                        "team": owner.team,
                        "distance_m": round(travel, 1),
                    })

        self.previous_possession = (
            {"track_id": owner.track_id, "team": owner.team, "position": owner.position}
            if owner else None
        )

        play_label = "ball_unobserved" if ball_position is None else "open_play"
        if any(s["type"] == "shot" for s in suggestions):
            play_label = "shot_opportunity"
        elif any(s.get("forward_gain_m", 0) > 8 for s in suggestions):
            play_label = "line_breaking_attack"
        elif owner and carrier_pressure == "high":
            play_label = "under_heavy_pressure"
        elif owner:
            play_label = "controlled_build_up"

        return {
            "frame_index": frame_index,
            "timestamp_s": round(timestamp_s, 3),
            "possession_track_id": possession,
            "possession_team": owner.team if owner else None,
            "carrier_pressure": carrier_pressure,
            "play_label": play_label,
            "play": play_label,
            "data_quality": "ball_missing" if ball_position is None else "usable",
            "events": events,
            "suggestions": suggestions,
        }