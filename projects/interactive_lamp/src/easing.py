from poses import Mood


def ease_in_out(t: float) -> float:
    """Smoothstep: slow-in, slow-out. Deliberate / weighted feel."""
    return t * t * (3 - 2 * t)


def ease_in_out_back(t: float) -> float:
    """Overshoots at both ends. Springy / happy.
    Returns values slightly below 0 near the start and above 1 near the end."""
    c1 = 1.70158  # larger = bouncier
    c2 = c1 * 1.525
    if t < 0.5:
        return (pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
    return (pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2


EASING = {
    Mood.NEUTRAL: ease_in_out,
    Mood.SAD: ease_in_out,
    Mood.HAPPY: ease_in_out_back,
}

DURATION_SCALE = {
    Mood.NEUTRAL: 1.0,
    Mood.SAD: 1.6,
    Mood.HAPPY: 0.6,
}
