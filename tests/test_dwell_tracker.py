import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dwell_tracker import Zone, DwellTracker


def make_tracker():
    # zona "perto do veículo": retângulo de (0,0) a (100,100)
    return DwellTracker(zone=Zone(0, 0, 100, 100))


def test_zone_contains_point_inside_and_outside():
    zone = Zone(0, 0, 100, 100)
    assert zone.contains((50, 50)) is True
    assert zone.contains((150, 50)) is False
    assert zone.contains((0, 0)) is True  # borda inclusiva


def test_person_entering_and_leaving_zone_creates_one_session():
    tracker = make_tracker()
    tracker.update([(1, 200, 200)], frame_idx=0)   # fora
    tracker.update([(1, 50, 50)], frame_idx=1)     # entra
    tracker.update([(1, 50, 50)], frame_idx=2)     # continua dentro
    tracker.update([(1, 200, 200)], frame_idx=3)   # sai
    tracker.finalize(last_frame_idx=3)

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].start_frame == 1
    assert sessions[0].end_frame == 3


def test_person_never_entering_zone_has_no_session():
    tracker = make_tracker()
    tracker.update([(1, 200, 200)], frame_idx=0)
    tracker.update([(1, 210, 200)], frame_idx=1)
    tracker.finalize(last_frame_idx=1)

    assert tracker.closed_sessions == []


def test_session_still_open_at_end_of_video_gets_closed_by_finalize():
    tracker = make_tracker()
    tracker.update([(1, 50, 50)], frame_idx=0)
    tracker.update([(1, 50, 50)], frame_idx=1)
    tracker.finalize(last_frame_idx=1)

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].end_frame == 1


def test_id_switch_while_inside_zone_continues_same_session():
    """Regressão do mesmo bug do contador-onibus: tracker perde o ID no
    meio da permanência e reatribui um novo — não pode virar 2 sessões."""
    tracker = make_tracker()
    tracker.update([(1, 50, 50)], frame_idx=0)
    tracker.update([(1, 50, 50)], frame_idx=1)
    tracker.update([], frame_idx=2)  # tracker perde o ID
    tracker.update([(2, 52, 52)], frame_idx=3)  # reaparece pertinho, ID novo
    tracker.update([(2, 200, 200)], frame_idx=4)  # sai da zona
    tracker.finalize(last_frame_idx=4)

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].track_id == 1  # mantém o ID canônico original
    assert sessions[0].start_frame == 0
    assert sessions[0].end_frame == 4


def test_duration_seconds_uses_fps():
    tracker = make_tracker()
    tracker.update([(1, 50, 50)], frame_idx=0)
    tracker.finalize(last_frame_idx=30)  # 30 frames a 15fps = 2s

    sessions = tracker.closed_sessions
    assert sessions[0].duration_seconds(fps=15) == 2.0


def test_two_people_get_independent_sessions():
    tracker = make_tracker()
    tracker.update([(1, 50, 50), (2, 60, 60)], frame_idx=0)
    tracker.update([(1, 200, 200), (2, 60, 60)], frame_idx=1)  # ID 1 sai, ID 2 continua dentro
    tracker.finalize(last_frame_idx=1)  # fecha a sessão de ID 2, ainda aberta

    sessions = {s.track_id: s for s in tracker.closed_sessions}
    assert set(sessions.keys()) == {1, 2}
    totals = tracker.total_seconds_per_id(fps=1)
    assert totals[1] == 1.0
    assert totals[2] == 1.0
