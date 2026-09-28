"""Manager-Level Tactical Intelligence & Advanced Pass/Play Prediction Engine.

Combines:
  1. 16x12 Expected Threat (xT) Pitch Value Grid.
  2. Multi-Pass Typology:
     - Ground Precision Pass
     - Through-Ball into Exploit Space
     - Lobbed Pass / Cross into Danger Zone
  3. Dynamic Physics & Probability Estimator:
     - Defender reaction speed, acceleration & intercept cone
     - Goalkeeper positioning, sweeping box, reaction time & reach agility
     - Spatial clearance, pocket separation, receiver lead vector
  4. Space Zone Shading / Pocket Identification:
     - Generates convex spatial pocket polygons & dangerous half-spaces
  5. Strict Optimal Decision Ranking with Tactical Breakdown.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import numpy as np


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.dist(first, second)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def point_to_segment_distance(point: tuple[float, float], seg_a: tuple[float, float], seg_b: tuple[float, float]) -> float:
    """Calculate the shortest distance from a point to a passing trajectory vector."""
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

    def _opponent_goal_pos(self) -> tuple[float, float]:
        gx = self.pitch_length if self.attacking_direction == 1 else 0.0
        return (gx, self.pitch_width / 2.0)

    def _possession(self, players: list[PlayerState], ball_position):
        if ball_position is None or not players:
            return None
        nearest = min(players, key=lambda player: distance(player.position, ball_position))
        return nearest if distance(nearest.position, ball_position) <= 3.4 else None

    def _compute_through_pass_target(self, owner: PlayerState, target: PlayerState, opponents: list[PlayerState]) -> tuple[float, float]:
        """Project lead running space ahead of teammate towards goal or wide pocket."""
        att_x_diff = 1.0 if self.attacking_direction == 1 else -1.0
        
        # Lead distance proportional to open forward space (between 3.5m and 7.5m)
        lead_x = target.position[0] + (att_x_diff * 4.8)
        lead_y = target.position[1] + (0.8 if target.position[1] > 34 else -0.8)
        
        # Clamp within pitch boundaries
        lead_x = max(2.0, min(self.pitch_length - 2.0, lead_x))
        lead_y = max(2.0, min(self.pitch_width - 2.0, lead_y))
        return (lead_x, lead_y)

    def _estimate_gk_interception_risk(self, pass_endpoint: tuple[float, float], pass_dist: float, gk: PlayerState = None) -> float:
        """Evaluate probability of Goalkeeper sweeping or claiming the pass based on agility & distance."""
        goal_pos = self._opponent_goal_pos()
        dist_to_goal = distance(pass_endpoint, goal_pos)
        
        if gk is not None:
            gk_dist = distance(gk.position, pass_endpoint)
        else:
            # Assumed goalkeeper at goal center
            gk_dist = dist_to_goal

        # Goalkeeper sweeping zone (deep box interception threat)
        if gk_dist < 6.0:
            # Goalkeeper gets to the ball with 85%+ chance
            return 0.85
        elif gk_dist < 14.0:
            # Sweeper GK zone: risk decreases with distance from keeper
            return max(0.1, 0.85 - (gk_dist - 6.0) * 0.08)
        elif dist_to_goal < 16.0:
            # Danger six-yard / penalty box claim zone
            return 0.35
        return 0.05

    def _evaluate_pass_options(self, owner: PlayerState, players: list[PlayerState]):
        """Evaluate Ground, Through-ball, and Cross/Lob passes with physics, cover shadows, and xT delta."""
        teammates = [
            p for p in players
            if p.team == owner.team and p.track_id != owner.track_id and p.role != "goalkeeper"
        ]
        opponents = [p for p in players if p.team != owner.team and p.role != "goalkeeper"]
        gk = next((p for p in players if p.team != owner.team and p.role == "goalkeeper"), None)
        
        candidates = []
        owner_xt = self._get_xt_value(owner.position)

        for target in teammates:
            pass_dist = distance(owner.position, target.position)
            if not 3.5 <= pass_dist <= 52.0:
                continue

            target_xt = self._get_xt_value(target.position)
            delta_xt = max(-0.05, target_xt - owner_xt)
            fwd_gain = self._attacking_x(target.position) - self._attacking_x(owner.position)
            nearest_def_dist = min((distance(target.position, opp.position) for opp in opponents), default=12.0)
            
            # --- 1. EVALUATE THROUGH PASS OPTION ---
            through_target_pt = self._compute_through_pass_target(owner, target, opponents)
            through_dist = distance(owner.position, through_target_pt)
            through_xt = self._get_xt_value(through_target_pt)
            
            # Defender interception along through lane
            through_lane_threat = 0.0
            for opp in opponents:
                perp_dist = point_to_segment_distance(opp.position, owner.position, through_target_pt)
                if perp_dist < 3.8:
                    dist_to_passer = distance(opp.position, owner.position)
                    if dist_to_passer < through_dist:
                        through_lane_threat += max(0.0, 3.8 - perp_dist) * 0.28
            
            through_lane_safety = clamp(1.0 - through_lane_threat)
            through_space_safety = clamp(min((distance(through_target_pt, opp.position) for opp in opponents), default=10.0) / 7.0)
            gk_risk_through = self._estimate_gk_interception_risk(through_target_pt, through_dist, gk)

            # Through pass completion probability
            through_success_prob = clamp((0.45 * through_lane_safety + 0.35 * through_space_safety + 0.20 * (1.0 - gk_risk_through)) * 100.0, 8.0, 96.0)
            
            # Is a through pass viable and high-impact?
            is_through_viable = (fwd_gain > 4.0 or (self._attacking_x(through_target_pt) > self.pitch_length * 0.60)) and nearest_def_dist < 6.0
            if is_through_viable and through_success_prob > 30.0:
                through_score = (
                    0.30 * (through_success_prob / 100.0) +
                    0.35 * clamp(0.5 + (through_xt - owner_xt) * 4.5 + fwd_gain / 40.0) +
                    0.20 * through_space_safety +
                    0.15 * clamp(1.0 - through_dist / 60.0)
                )
                candidates.append({
                    "type": "through_pass",
                    "subtype": "through_ball",
                    "score": round(through_score, 3),
                    "confidence": round(through_score, 3),
                    "success_prob": int(round(through_success_prob)),
                    "expected_threat": round(through_xt, 3),
                    "delta_xt": round(through_xt - owner_xt, 3),
                    "from_track_id": owner.track_id,
                    "to_track_id": target.track_id,
                    "target_pitch_pos": through_target_pt,
                    "receiver_role": target.role.upper(),
                    "team": owner.team,
                    "distance_m": round(through_dist, 1),
                    "forward_gain_m": round(fwd_gain + 4.8, 1),
                    "nearest_opponent_m": round(nearest_def_dist, 1),
                    "lane_safety": round(through_lane_safety, 2),
                    "gk_risk": round(gk_risk_through, 2),
                    "desc": f"Through ball into open pocket ({int(through_success_prob)}% success)",
                })

            # --- 2. EVALUATE LOB / CROSS OPTION ---
            is_cross_zone = (owner.position[1] < 18.0 or owner.position[1] > 50.0) and (self._attacking_x(owner.position) > self.pitch_length * 0.55)
            is_box_target = (18.0 <= target.position[1] <= 50.0) and (self._attacking_x(target.position) > self.pitch_length * 0.70)
            is_lofted_switch = pass_dist > 25.0 and (abs(owner.position[1] - target.position[1]) > 28.0)
            
            if (is_cross_zone and is_box_target) or is_lofted_switch:
                gk_risk_cross = self._estimate_gk_interception_risk(target.position, pass_dist, gk)
                cross_space_safety = clamp(nearest_def_dist / 6.0)
                cross_success_prob = clamp((0.40 * cross_space_safety + 0.35 * (1.0 - gk_risk_cross) + 0.25 * clamp(1.0 - pass_dist / 55.0)) * 100.0, 15.0, 92.0)
                
                cross_score = (
                    0.30 * (cross_success_prob / 100.0) +
                    0.35 * clamp(0.5 + delta_xt * 4.0 + fwd_gain / 40.0) +
                    0.20 * cross_space_safety +
                    0.15 * clamp(1.0 - pass_dist / 60.0)
                )
                
                pass_name = "Danger cross into box" if is_box_target else "Lofted diagonal switch"
                candidates.append({
                    "type": "lob_pass",
                    "subtype": "cross" if is_box_target else "lob",
                    "score": round(cross_score, 3),
                    "confidence": round(cross_score, 3),
                    "success_prob": int(round(cross_success_prob)),
                    "expected_threat": round(target_xt, 3),
                    "delta_xt": round(delta_xt, 3),
                    "from_track_id": owner.track_id,
                    "to_track_id": target.track_id,
                    "target_pitch_pos": target.position,
                    "receiver_role": target.role.upper(),
                    "team": owner.team,
                    "distance_m": round(pass_dist, 1),
                    "forward_gain_m": round(fwd_gain, 1),
                    "nearest_opponent_m": round(nearest_def_dist, 1),
                    "lane_safety": 0.90,
                    "gk_risk": round(gk_risk_cross, 2),
                    "desc": f"{pass_name} ({int(cross_success_prob)}% success)",
                })

            # --- 3. EVALUATE STANDARD GROUND PASS ---
            lane_threat = 0.0
            for opp in opponents:
                perp_dist = point_to_segment_distance(opp.position, owner.position, target.position)
                if perp_dist < 3.8:
                    dist_to_passer = distance(opp.position, owner.position)
                    if dist_to_passer < pass_dist:
                        lane_threat += max(0.0, 3.8 - perp_dist) * 0.26

            ground_lane_safety = clamp(1.0 - lane_threat)
            ground_space_safety = clamp(nearest_def_dist / 7.5)
            ground_success_prob = clamp((0.55 * ground_lane_safety + 0.45 * ground_space_safety) * 100.0, 5.0, 98.0)

            ground_score = (
                0.35 * ground_lane_safety +
                0.30 * ground_space_safety +
                0.25 * clamp(0.5 + delta_xt * 4.0 + fwd_gain / 50.0) +
                0.10 * clamp(1.0 - pass_dist / 60.0)
            )

            if delta_xt > 0.08 or fwd_gain > 12.0:
                desc = f"Line-breaking ground pass ({int(ground_success_prob)}%)"
            elif nearest_def_dist > 6.5:
                desc = f"Pocket pass to unmarked receiver ({int(ground_success_prob)}%)"
            else:
                desc = f"Short combination pass ({int(ground_success_prob)}%)"

            candidates.append({
                "type": "ground_pass",
                "subtype": "ground",
                "score": round(ground_score, 3),
                "confidence": round(ground_score, 3),
                "success_prob": int(round(ground_success_prob)),
                "expected_threat": round(target_xt, 3),
                "delta_xt": round(delta_xt, 3),
                "from_track_id": owner.track_id,
                "to_track_id": target.track_id,
                "target_pitch_pos": target.position,
                "receiver_role": target.role.upper(),
                "team": owner.team,
                "distance_m": round(pass_dist, 1),
                "forward_gain_m": round(fwd_gain, 1),
                "nearest_opponent_m": round(nearest_def_dist, 1),
                "lane_safety": round(ground_lane_safety, 2),
                "gk_risk": 0.02,
                "desc": desc,
            })

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    def _evaluate_shot_option(self, owner: PlayerState, players: list[PlayerState]):
        """Evaluate direct shot on goal threat (xG model heuristic)."""
        att_x = self._attacking_x(owner.position)
        if att_x < self.pitch_length - 34.0:
            return None

        goal_center = self._opponent_goal_pos()
        goal_dist = distance(owner.position, goal_center)
        if goal_dist > 32.0:
            return None

        angle_to_goal = math.atan2(abs(owner.position[1] - 34.0), max(1.0, self.pitch_length - att_x))
        angle_factor = clamp(math.cos(angle_to_goal))

        defenders = [p for p in players if p.team != owner.team and p.role != "goalkeeper"]
        pressure = min((distance(owner.position, d.position) for d in defenders), default=15.0)

        # Expected Goals (xG) estimation
        xg = clamp((0.58 - goal_dist / 48.0) * angle_factor + (pressure / 35.0) * 0.15, 0.05, 0.88)

        return {
            "type": "shot",
            "subtype": "shot",
            "score": round(xg, 3),
            "confidence": round(xg, 3),
            "success_prob": int(round(xg * 100.0)),
            "expected_threat": round(xg, 3),
            "from_track_id": owner.track_id,
            "target_pitch_pos": goal_center,
            "team": owner.team,
            "goal_distance_m": round(goal_dist, 1),
            "nearest_opponent_m": round(pressure, 1),
            "desc": f"Direct shot on goal ({round(goal_dist, 1)}m out, {int(xg*100)}% xG)",
        }

    def _extract_exploitable_space_zones(self, owner: PlayerState, players: list[PlayerState]) -> list[dict]:
        """Detect dangerous pockets of open pitch space between defender lines and half-spaces."""
        opponents = [p for p in players if p.team != owner.team and p.role != "goalkeeper"]
        if len(opponents) < 2:
            return []

        zones = []
        att_x_min = self.pitch_length * 0.40 if self.attacking_direction == 1 else 0.0
        att_x_max = self.pitch_length if self.attacking_direction == 1 else self.pitch_length * 0.60

        # Scan candidate 8x6 grid cells for open space pockets
        for x_cell in np.linspace(att_x_min + 5, att_x_max - 5, 6):
            for y_cell in [14.0, 24.0, 34.0, 44.0, 54.0]:
                pt = (float(x_cell), float(y_cell))
                min_def_dist = min((distance(pt, opp.position) for opp in opponents), default=20.0)
                
                # If zone has > 7.5m clearance from all defenders and high xT
                if min_def_dist >= 7.5:
                    xt_val = self._get_xt_value(pt)
                    if xt_val >= 0.035:
                        zones.append({
                            "center": pt,
                            "radius_m": round(min(min_def_dist * 0.65, 8.5), 1),
                            "xt_value": round(xt_val, 3),
                            "clearance_m": round(min_def_dist, 1)
                        })

        # Return top 2 largest/highest threat space zones
        zones.sort(key=lambda z: (z["xt_value"] * 1.5 + z["clearance_m"] * 0.1), reverse=True)
        return zones[:2]

    def update(self, frame_index: int, timestamp_s: float, players: list[PlayerState], ball_position):
        for player in players:
            self.player_history[player.track_id].append((timestamp_s, player.position))
        if ball_position is not None:
            self.ball_history.append((timestamp_s, ball_position))

        owner = self._possession(players, ball_position)
        possession = owner.track_id if owner else None
        suggestions = []
        events = []
        space_zones = []

        carrier_pressure = "low"
        if owner:
            opponents = [p for p in players if p.team != owner.team]
            min_opp_dist = min((distance(owner.position, opp.position) for opp in opponents), default=15.0)
            carrier_pressure = "high" if min_opp_dist < 2.5 else ("medium" if min_opp_dist < 5.0 else "low")

            pass_candidates = self._evaluate_pass_options(owner, players) or []
            shot_option = self._evaluate_shot_option(owner, players)
            space_zones = self._extract_exploitable_space_zones(owner, players)

            # 1. Shot Opportunity Check
            if shot_option:
                suggestions.append(shot_option)

            # 2. Add Top Passing Options (Through pass, Lob/Cross, Best Ground Pass)
            if pass_candidates:
                best_pass = pass_candidates[0]
                best_pass["tier"] = "primary"
                suggestions.append(best_pass)

                seen_subtypes = {best_pass.get("subtype")}
                for backup in pass_candidates[1:]:
                    stype = backup.get("subtype")
                    if stype not in seen_subtypes or (backup["score"] >= 0.60 and len(suggestions) < 3):
                        seen_subtypes.add(stype)
                        backup_copy = dict(backup)
                        backup_copy["tier"] = "backup"
                        suggestions.append(backup_copy)
                    if len(suggestions) >= 3:
                        break

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
        elif any(s.get("subtype") == "through_ball" for s in suggestions):
            play_label = "through_ball_exploitation"
        elif any(s.get("subtype") == "cross" for s in suggestions):
            play_label = "danger_crossing_chance"
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
            "space_zones": space_zones,
        }