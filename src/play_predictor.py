"""Heuristic football event and next-play suggestions from tracked states.

This is an interpretable baseline, not a trained event classifier. It creates
training data and a useful live suggestion stream until event labels are
available for a learned model.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import math


def distance(first, second) -> float:
    return math.dist(first, second)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class PlayerState:
    track_id: int
    team: int
    role: str
    position: tuple[float, float]


class PlayPredictor:
    """Track short-term ball/player state and produce interpretable signals."""

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0, attacking_direction: int = 1):
        if attacking_direction not in (-1, 1):
            raise ValueError("attacking_direction must be -1 or 1")
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.attacking_direction = attacking_direction
        self.player_history = defaultdict(lambda: deque(maxlen=8))
        self.ball_history = deque(maxlen=12)
        self.previous_possession = None

    def _attacking_x(self, position: tuple[float, float]) -> float:
        return position[0] if self.attacking_direction == 1 else self.pitch_length - position[0]

    def _possession(self, players: list[PlayerState], ball_position):
        if ball_position is None or not players:
            return None
        nearest = min(players, key=lambda player: distance(player.position, ball_position))
        return nearest if distance(nearest.position, ball_position) <= 3.0 else None

    def _pass_suggestion(self, owner: PlayerState, players: list[PlayerState]):
        teammates = [
            player for player in players
            if player.team == owner.team and player.track_id != owner.track_id and player.role != "goalkeeper"
        ]
        candidates = []
        for target in teammates:
            pass_distance = distance(owner.position, target.position)
            if not 6.0 <= pass_distance <= 32.0:
                continue
            opponents = [player for player in players if player.team != owner.team]
            nearest_opponent = min(
                (distance(target.position, opponent.position) for opponent in opponents),
                default=10.0,
            )
            forward_gain = self._attacking_x(target.position) - self._attacking_x(owner.position)
            score = clamp(0.45 + forward_gain / 40.0 + nearest_opponent / 30.0 - pass_distance / 100.0)
            candidates.append((score, target, nearest_opponent, forward_gain))
        if not candidates:
            return None
        score, target, nearest_opponent, forward_gain = max(candidates, key=lambda item: item[0])
        return {
            "type": "pass",
            "confidence": round(score, 3),
            "from_track_id": owner.track_id,
            "to_track_id": target.track_id,
            "team": owner.team,
            "distance_m": round(distance(owner.position, target.position), 2),
            "forward_gain_m": round(forward_gain, 2),
            "nearest_opponent_m": round(nearest_opponent, 2),
        }

    def _shot_suggestion(self, owner: PlayerState, players: list[PlayerState]):
        attacking_x = self._attacking_x(owner.position)
        if attacking_x < self.pitch_length - 30.0:
            return None
        goal = (self.pitch_length if self.attacking_direction == 1 else 0.0, self.pitch_width / 2.0)
        goal_distance = distance(owner.position, goal)
        if goal_distance > 30.0:
            return None
        defenders = [player for player in players if player.team != owner.team]
        pressure = min((distance(owner.position, defender.position) for defender in defenders), default=15.0)
        confidence = clamp(0.8 - goal_distance / 80.0 + pressure / 60.0)
        return {
            "type": "shot",
            "confidence": round(confidence, 3),
            "from_track_id": owner.track_id,
            "team": owner.team,
            "goal_distance_m": round(goal_distance, 2),
            "nearest_opponent_m": round(pressure, 2),
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

        if owner:
            pass_suggestion = self._pass_suggestion(owner, players)
            if pass_suggestion:
                suggestions.append(pass_suggestion)
            shot_suggestion = self._shot_suggestion(owner, players)
            if shot_suggestion:
                suggestions.append(shot_suggestion)

        if owner and self.previous_possession and owner.track_id != self.previous_possession["track_id"]:
            previous = self.previous_possession
            if owner.team == previous["team"] and previous["position"] is not None:
                travel = distance(previous["position"], ball_position) if ball_position else 0.0
                if travel >= 4.0:
                    events.append({
                        "type": "completed_pass",
                        "confidence": round(clamp(0.55 + travel / 30.0), 3),
                        "from_track_id": previous["track_id"],
                        "to_track_id": owner.track_id,
                        "team": owner.team,
                        "distance_m": round(travel, 2),
                    })

        self.previous_possession = (
            {"track_id": owner.track_id, "team": owner.team, "position": owner.position}
            if owner else None
        )
        play = "ball_unobserved" if ball_position is None else "possession_unknown"
        if any(item["type"] == "shot" for item in suggestions):
            play = "attacking_threat"
        elif owner and any(item["type"] == "pass" and item["forward_gain_m"] > 4 for item in suggestions):
            play = "progressive_pass"
        elif owner:
            play = "in_possession"

        return {
            "frame_index": frame_index,
            "timestamp_s": round(timestamp_s, 3),
            "possession_track_id": possession,
            "possession_team": owner.team if owner else None,
            "play": play,
            "data_quality": "ball_missing" if ball_position is None else "usable",
            "events": events,
            "suggestions": suggestions,
        }