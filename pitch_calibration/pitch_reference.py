"""
Standard football pitch landmark coordinates, in metres.
------------------------------------------------------------
Origin (0, 0) is the top-left corner of the pitch as viewed from directly
above, with x running along the length (0 -> PITCH_LENGTH) and y along the
width (0 -> PITCH_WIDTH).

Dimensions default to a standard 105m x 68m pitch (common for professional
matches). If your footage is from a pitch with different dimensions, adjust
PITCH_LENGTH / PITCH_WIDTH below -- everything else derives from those two
numbers, using standard football pitch proportions (penalty box, six-yard
box, center circle, penalty spot placement all follow fixed rules
regardless of overall pitch size).

You do NOT need every landmark visible in a given shot -- calibrate.py lets
you pick whichever subset is visible in your frame. You need at least 4
non-collinear points for a stable homography; more (and more spread out)
is better.
"""

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

PENALTY_BOX_LENGTH = 16.5    # depth from goal line
PENALTY_BOX_WIDTH = 40.32    # full width
SIX_YARD_BOX_LENGTH = 5.5
SIX_YARD_BOX_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CENTER_CIRCLE_RADIUS = 9.15

_pb_y0 = (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2
_pb_y1 = _pb_y0 + PENALTY_BOX_WIDTH
_sb_y0 = (PITCH_WIDTH - SIX_YARD_BOX_WIDTH) / 2
_sb_y1 = _sb_y0 + SIX_YARD_BOX_WIDTH

# Every landmark below is a named point you can click on in calibrate.py.
# Names are deliberately explicit (left/right = along the pitch length as
# broadcast typically shows it, but it's just a label -- what matters is
# that the (x, y) values are geometrically correct).
PITCH_LANDMARKS = {
    # Corners
    "corner_top_left": (0.0, 0.0),
    "corner_top_right": (PITCH_LENGTH, 0.0),
    "corner_bottom_right": (PITCH_LENGTH, PITCH_WIDTH),
    "corner_bottom_left": (0.0, PITCH_WIDTH),

    # Halfway line
    "halfway_top": (PITCH_LENGTH / 2, 0.0),
    "halfway_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH),
    "center_spot": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),

    # Left penalty box (near x=0 goal)
    "left_penalty_box_top_left": (0.0, _pb_y0),
    "left_penalty_box_top_right": (PENALTY_BOX_LENGTH, _pb_y0),
    "left_penalty_box_bottom_right": (PENALTY_BOX_LENGTH, _pb_y1),
    "left_penalty_box_bottom_left": (0.0, _pb_y1),
    "left_six_yard_top_left": (0.0, _sb_y0),
    "left_six_yard_top_right": (SIX_YARD_BOX_LENGTH, _sb_y0),
    "left_six_yard_bottom_right": (SIX_YARD_BOX_LENGTH, _sb_y1),
    "left_six_yard_bottom_left": (0.0, _sb_y1),
    "left_penalty_spot": (PENALTY_SPOT_DISTANCE, PITCH_WIDTH / 2),

    # Right penalty box (near x=PITCH_LENGTH goal)
    "right_penalty_box_top_left": (PITCH_LENGTH - PENALTY_BOX_LENGTH, _pb_y0),
    "right_penalty_box_top_right": (PITCH_LENGTH, _pb_y0),
    "right_penalty_box_bottom_right": (PITCH_LENGTH, _pb_y1),
    "right_penalty_box_bottom_left": (PITCH_LENGTH - PENALTY_BOX_LENGTH, _pb_y1),
    "right_six_yard_top_left": (PITCH_LENGTH - SIX_YARD_BOX_LENGTH, _sb_y0),
    "right_six_yard_top_right": (PITCH_LENGTH, _sb_y0),
    "right_six_yard_bottom_right": (PITCH_LENGTH, _sb_y1),
    "right_six_yard_bottom_left": (PITCH_LENGTH - SIX_YARD_BOX_LENGTH, _sb_y1),
    "right_penalty_spot": (PITCH_LENGTH - PENALTY_SPOT_DISTANCE, PITCH_WIDTH / 2),
}