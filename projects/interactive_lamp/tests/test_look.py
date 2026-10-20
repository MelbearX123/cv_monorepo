from fakes import FakeVision
from perception.look import LookBehavior
from perception.vision_brain import Detection

CUP_EARLY = Detection("cup", 0.9, (0, 0, 10, 10))
CUP_LATE = Detection("cup", 0.8, (50, 50, 10, 10))
BOOK = Detection("book", 0.95, (0, 0, 10, 10))


def run(look: LookBehavior, frame, steps: int) -> None:
    for _ in range(steps):
        look.step(frame)


def test_idle_until_started(frame):
    look = LookBehavior(FakeVision(main_objects=[CUP_EARLY]), frames=1)
    run(look, frame, 3)
    assert not look.active
    assert look.take_result() is None


def test_majority_label_wins_and_latest_sighting_is_remembered(frame):
    vision = FakeVision(main_objects=[CUP_EARLY, BOOK, None, CUP_LATE])
    look = LookBehavior(vision, frames=4)
    look.start()

    run(look, frame, 3)
    assert look.active
    assert look.take_result() is None

    run(look, frame, 1)
    assert not look.active
    assert look.take_result().label == "cup"
    assert vision.remembered == [CUP_LATE]


def test_result_is_delivered_once(frame):
    look = LookBehavior(FakeVision(main_objects=[CUP_EARLY]), frames=1)
    look.start()
    run(look, frame, 1)
    assert look.take_result() is not None
    assert look.take_result() is None


def test_nothing_seen_reports_empty_label(frame):
    vision = FakeVision()
    look = LookBehavior(vision, frames=2)
    look.start()
    run(look, None, 2)
    assert look.take_result().label is None
    assert vision.remembered == []


def test_cancel_drops_votes_and_result(frame):
    look = LookBehavior(FakeVision(main_objects=[CUP_EARLY, CUP_EARLY]), frames=2)
    look.start()
    run(look, frame, 1)
    look.cancel()
    run(look, frame, 1)
    assert not look.active
    assert look.take_result() is None


def test_restart_clears_previous_votes(frame):
    look = LookBehavior(FakeVision(main_objects=[BOOK, CUP_EARLY]), frames=2)
    look.start()
    run(look, frame, 1)
    look.start()
    run(look, frame, 2)
    assert look.take_result().label == "cup"
