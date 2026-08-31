import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dwell_tracker import Zone, DwellTracker, DwellSession, filter_spurious_sessions


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


def test_session_closes_on_its_own_when_person_disappears_for_good():
    """Regressão de um bug real achado testando com vídeo de verdade: pessoa
    entra na zona e depois some pra sempre (sai de cena, oclusão longa) —
    a sessão não pode ficar "aberta" até o fim do vídeo inteiro, inflando
    a duração registrada muito além do tempo em que ela foi vista de fato."""
    tracker = DwellTracker(zone=Zone(0, 0, 100, 100), max_absence_frames=5)
    tracker.update([(1, 50, 50)], frame_idx=0)
    tracker.update([(1, 50, 50)], frame_idx=1)
    # ID 1 some de vez a partir daqui — nunca mais reaparece
    for frame_idx in range(2, 20):
        tracker.update([], frame_idx=frame_idx)
    tracker.finalize(last_frame_idx=100)  # vídeo "continua" por muito mais tempo

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].end_frame == 1  # fecha no último frame em que foi visto de verdade
    assert sessions[0].end_frame != 100  # não infla até o fim do vídeo


def test_brief_absence_within_tolerance_does_not_close_session_early():
    tracker = DwellTracker(zone=Zone(0, 0, 100, 100), max_absence_frames=5)
    tracker.update([(1, 50, 50)], frame_idx=0)
    tracker.update([], frame_idx=1)  # some por só 1 frame, dentro da tolerância
    tracker.update([(1, 50, 50)], frame_idx=2)
    tracker.update([(1, 200, 200)], frame_idx=3)  # sai da zona de verdade
    tracker.finalize(last_frame_idx=3)

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].start_frame == 0
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


def test_filter_spurious_sessions_drops_short_ones():
    """Caso real: 4 IDs fantasma de <0.5s (oclusão parcial) junto com a
    pessoa de verdade de 7.8s — só a sessão real deve sobrar."""
    sessions = [
        DwellSession(track_id=1, start_frame=0, end_frame=234),   # 7.8s a 30fps
        DwellSession(track_id=2, start_frame=45, end_frame=45),   # 0.0s (fantasma)
        DwellSession(track_id=3, start_frame=48, end_frame=60),   # 0.4s (fantasma)
        DwellSession(track_id=4, start_frame=96, end_frame=105),  # 0.3s (fantasma)
        DwellSession(track_id=5, start_frame=282, end_frame=291), # 0.3s (fantasma)
    ]
    filtrado = filter_spurious_sessions(sessions, fps=30, min_seconds=1.0)
    assert [s.track_id for s in filtrado] == [1]


def test_filter_spurious_sessions_keeps_sessions_at_or_above_threshold():
    sessions = [
        DwellSession(track_id=1, start_frame=0, end_frame=30),  # exatamente 1.0s a 30fps
        DwellSession(track_id=2, start_frame=0, end_frame=29),  # pouco menos de 1.0s
    ]
    filtrado = filter_spurious_sessions(sessions, fps=30, min_seconds=1.0)
    assert [s.track_id for s in filtrado] == [1]


def test_filter_spurious_sessions_default_threshold_is_one_second():
    sessions = [
        DwellSession(track_id=1, start_frame=0, end_frame=15),  # 0.5s a 30fps
        DwellSession(track_id=2, start_frame=0, end_frame=45),  # 1.5s a 30fps
    ]
    filtrado = filter_spurious_sessions(sessions, fps=30)
    assert [s.track_id for s in filtrado] == [2]


def test_filter_spurious_sessions_empty_list_returns_empty():
    assert filter_spurious_sessions([], fps=30) == []


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
