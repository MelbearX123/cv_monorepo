from enum import StrEnum

from body.poses import Mood


class Category(StrEnum):
    GRATITUDE = "gratitude"
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class PointOutcome(StrEnum):
    FOUND = "found"
    OFF = "off"
    MISSING = "missing"


KEYWORDS: dict[Category, set[str]] = {
    Category.GRATITUDE: {"thanks", "thank", "appreciate", "grateful"},
    Category.AFFIRMATIVE: {
        "yes",
        "yeah",
        "yep",
        "sure",
        "good",
        "great",
        "fine",
        "okay",
        "ok",
        "better",
        "happy",
    },
    Category.NEGATIVE: {
        "no",
        "nope",
        "not",
        "tired",
        "stressed",
        "rough",
        "bad",
        "sad",
        "down",
        "awful",
        "terrible",
        "exhausted",
    },
}

REPLIES: dict[Category, list[str]] = {
    Category.GRATITUDE: [
        "Anytime.",
        "Happy to help.",
        "Of course.",
    ],
    Category.AFFIRMATIVE: [
        "Glad to hear it.",
        "That's good to hear.",
        "Love that.",
    ],
    Category.NEGATIVE: [
        "I'm sorry to hear that.",
        "That sounds tough. I'm here.",
        "Rough one. Take it easy.",
    ],
    Category.UNKNOWN: [
        "I hear you.",
        "Got it.",
        "Thanks for telling me.",
    ],
}

LOOK_KEYWORDS: set[str] = {"look"}
LOOK_PHRASES: tuple[str, ...] = ("check this out",)
LOOK_REPLIES: list[str] = [
    "Oh, {article} {label}! Nice.",
    "That looks like {article} {label}.",
    "I see {article} {label}.",
    "Is that {article} {label}?",
]
LOOK_EMPTY: list[str] = [
    "I can't quite make out what you're showing me.",
    "Hmm, I don't see anything I recognize.",
    "Show me a little closer?",
]

RECALL_PHRASES: tuple[str, ...] = (
    "have you seen",
    "did you see",
    "seen my",
    "where is",
    "where's",
)
RECALL_YES: list[str] = [
    "Yes! You showed me {article} {label} at {when} — it was over here!",
    "I did. I saw {article} {label} around {when}, right over here!",
    "Yep, {article} {label} at {when}. It was over here!",
]
RECALL_NO: list[str] = [
    "I don't think you've shown me that recently.",
    "Hmm, I haven't seen that lately.",
    "Not that I recall seeing.",
]

POINT_PHRASES: tuple[str, ...] = (
    "point at",
    "point to",
    "point out",
    "can you find",
    "find the",
    "find my",
    "where's the",
)
POINT_REPLIES: dict[PointOutcome, list[str]] = {
    PointOutcome.FOUND: [
        "There it is — {article} {label}, right there!",
        "Found it! The {label} is over there.",
        "That {label}? It's right here.",
    ],
    PointOutcome.OFF: [
        "I think the {label} is somewhere over there.",
        "The {label}'s around there, roughly.",
    ],
    PointOutcome.MISSING: [
        "I don't see {article} {label} right now.",
        "Hmm, no {label} in view at the moment.",
        "I can't spot {article} {label} here.",
    ],
}

MOOD_OPENINGS: dict[Mood, list[str]] = {
    Mood.HAPPY: [
        "You look cheerful today!",
        "Someone's in a good mood.",
    ],
    Mood.SAD: [
        "You seem a little down. Everything okay?",
        "Rough day? I'm here if you want to talk.",
    ],
}
