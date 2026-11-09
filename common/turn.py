"""Is a detected face turned away from the camera? Used by mask_detection and beard_detection only.

YuNet gives 5 landmarks per face: right eye, left eye, nose, right mouth corner, left mouth corner.
Facing the camera, the nose sits halfway between the eyes and the eyes are far apart. As the head turns, the nose
slides towards one eye and the eyes move closer together.
"""


def turn_amount(row):
    """(nose offset, eye distance ratio). Frontal: about (0.05-0.2, 0.40-0.50). Side view: (0.5+, below 0.25)."""
    re_x, le_x, nose_x, w = float(row[4]), float(row[6]), float(row[8]), max(float(row[2]), 1.0)
    eye_dist = abs(le_x - re_x)
    if eye_dist < 1.0:
        return 9.0, 0.0
    return abs(nose_x - (re_x + le_x) / 2.0) / eye_dist, eye_dist / w


def is_turned(row, max_turn=0.35, min_eye_ratio=0.28):
    t, r = turn_amount(row)
    return t > max_turn or r < min_eye_ratio
