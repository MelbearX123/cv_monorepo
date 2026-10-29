import pytest

from dialogue.intent import Intent, classify


@pytest.mark.parametrize(
    "text, intent",
    [
        ("play some music", Intent.PLAY_MUSIC),
        ("music please", Intent.PLAY_MUSIC),
        ("stop", Intent.STOP_MUSIC),
        ("turn it off", Intent.STOP_MUSIC),
        ("point at the cup", Intent.POINT),
        ("can you find my phone", Intent.POINT),
        ("have you seen my keys", Intent.RECALL),
        ("did you see the bottle", Intent.RECALL),
        ("look at this", Intent.LOOK),
        ("check this out", Intent.LOOK),
        ("I had a nice day", Intent.CONVERSE),
        ("", Intent.CONVERSE),
    ],
)
def test_classify(text, intent):
    assert classify(text) == intent


def test_classify_ignores_case():
    assert classify("PLAY SOME MUSIC") == Intent.PLAY_MUSIC


def test_stop_wins_over_play():
    assert classify("stop playing some music") == Intent.STOP_MUSIC


def test_wheres_the_points_but_wheres_my_recalls():
    assert classify("where's the cup") == Intent.POINT
    assert classify("where's my cup") == Intent.RECALL


def test_look_matches_whole_words_only():
    assert classify("you're looking good") == Intent.CONVERSE
