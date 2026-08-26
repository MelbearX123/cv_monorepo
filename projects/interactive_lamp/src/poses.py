from enum import StrEnum


class Joints(StrEnum):
    BASE_YAW = "base_yaw_joint"
    SHOULDER_PITCH = "shoulder_pitch_joint"
    ELBOW_PITCH = "elbow_pitch_joint"
    NECK_YAW = "neck_yaw_joint"
    HEAD_PITCH = "head_pitch_joint"


class Mood(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"


poses = {
    "engage": {
        "joints": {
            Joints.BASE_YAW: 0,
            Joints.SHOULDER_PITCH: 0.2,
            Joints.ELBOW_PITCH: -0.9,
            Joints.NECK_YAW: 0,
            Joints.HEAD_PITCH: -0.3,
        },
        "light": [1.0, 1.0, 1.0],  # bright white
    },
    "disengage": {
        "joints": {
            Joints.BASE_YAW: 0,
            Joints.SHOULDER_PITCH: -0.2,
            Joints.ELBOW_PITCH: -0.7,
            Joints.NECK_YAW: 0,
            Joints.HEAD_PITCH: -0.1,
        },
        "light": [0.2, 0.2, 0.2],  # dim white
    },
    "nod_up": {
        "joints": {
            Joints.BASE_YAW: 0,
            Joints.SHOULDER_PITCH: 0.2,
            Joints.ELBOW_PITCH: -0.9,
            Joints.NECK_YAW: 0,
            Joints.HEAD_PITCH: -0.6,
        },
        "light": [0.0, 1.0, 0.0],  # green
    },
    "nod_down": {
        "joints": {
            Joints.BASE_YAW: 0,
            Joints.SHOULDER_PITCH: 0.2,
            Joints.ELBOW_PITCH: -0.9,
            Joints.NECK_YAW: 0,
            Joints.HEAD_PITCH: 0,
        },
        "light": [0.0, 1.0, 0.0],  # green
    },
    "shake_left": {
        "joints": {
            Joints.BASE_YAW: 0,
            Joints.SHOULDER_PITCH: 0.2,
            Joints.ELBOW_PITCH: -0.9,
            Joints.NECK_YAW: -0.5,
            Joints.HEAD_PITCH: -0.3,
        },
        "light": [1.0, 0.0, 0.0],  # red
    },
    "shake_right": {
        "joints": {
            Joints.BASE_YAW: 0,
            Joints.SHOULDER_PITCH: 0.2,
            Joints.ELBOW_PITCH: -0.9,
            Joints.NECK_YAW: 0.5,
            Joints.HEAD_PITCH: -0.3,
        },
        "light": [1.0, 0.0, 0.0],  # red
    },
}

gestures = {
    "engage": ["engage"],
    "disengage": ["disengage"],
    "nod": ["nod_down", "nod_up", "nod_down"],
    "shake": ["shake_left", "shake_right", "shake_left"],
}
