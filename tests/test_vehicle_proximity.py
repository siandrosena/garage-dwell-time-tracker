import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vehicle_proximity import BoundingBox, is_near_vehicle, closest_vehicle, VehicleProximityTracker


def test_person_inside_vehicle_box_is_near():
    vehicle = BoundingBox(100, 100, 200, 150)
    person = BoundingBox(140, 110, 160, 140)  # centro dentro do carro
    assert is_near_vehicle(person, vehicle) is True


def test_person_far_from_vehicle_is_not_near():
    vehicle = BoundingBox(100, 100, 200, 150)
    person = BoundingBox(500, 500, 520, 560)
    assert is_near_vehicle(person, vehicle) is False


def test_person_just_outside_box_but_within_margin_is_near():
    # carro de 100px de largura, margem 0.5 = 50px de folga pra cada lado
    vehicle = BoundingBox(100, 100, 200, 150)
    person_perto = BoundingBox(230, 115, 250, 135)  # centro em x=240, 40px além da borda (200+50=250 é o limite)
    assert is_near_vehicle(person_perto, vehicle, margin_ratio=0.5) is True


def test_bigger_vehicle_has_bigger_margin_same_pixel_distance():
    """Mesma distância em pixels da borda do veículo: perto pra um caminhão
    grande, longe pra um carro pequeno — a margem escala com o tamanho do veículo."""
    carro_pequeno = BoundingBox(100, 100, 150, 130)  # 50px de largura
    caminhao_grande = BoundingBox(100, 100, 300, 180)  # 200px de largura

    pessoa = BoundingBox(170, 110, 190, 125)  # centro em x=180, 30px além da borda dos 2

    assert is_near_vehicle(pessoa, carro_pequeno, margin_ratio=0.5) is False  # margem de só 25px
    assert is_near_vehicle(pessoa, caminhao_grande, margin_ratio=0.5) is True  # margem de 100px


def test_closest_vehicle_picks_nearest_among_multiple():
    pessoa = BoundingBox(95, 95, 105, 105)  # centro em (100,100)
    veiculos = {
        "onibus_A": BoundingBox(0, 0, 50, 50),      # longe
        "onibus_B": BoundingBox(90, 90, 130, 130),  # perto
    }
    assert closest_vehicle(pessoa, veiculos) == "onibus_B"


def test_closest_vehicle_returns_none_when_nothing_near():
    pessoa = BoundingBox(1000, 1000, 1010, 1010)
    veiculos = {"onibus_A": BoundingBox(0, 0, 50, 50)}
    assert closest_vehicle(pessoa, veiculos) is None


def test_proximity_session_opens_and_closes_with_vehicle():
    tracker = VehicleProximityTracker()
    onibus = {"1048": BoundingBox(100, 100, 200, 150)}

    tracker.update({1: BoundingBox(500, 500, 520, 520)}, onibus, frame_idx=0)  # longe
    tracker.update({1: BoundingBox(140, 110, 160, 130)}, onibus, frame_idx=1)  # chega perto
    tracker.update({1: BoundingBox(140, 110, 160, 130)}, onibus, frame_idx=2)  # continua
    tracker.update({1: BoundingBox(500, 500, 520, 520)}, onibus, frame_idx=3)  # se afasta
    tracker.finalize(last_frame_idx=3)

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].person_id == 1
    assert sessions[0].vehicle_id == "1048"
    assert sessions[0].start_frame == 1
    assert sessions[0].end_frame == 3


def test_person_walking_past_without_stopping_creates_short_session():
    """Passar andando gera uma sessão CURTA — é o combo com filter_spurious_sessions
    (do dwell_tracker) que separa 'passou perto' de 'ficou trabalhando'."""
    tracker = VehicleProximityTracker()
    onibus = {"1048": BoundingBox(100, 100, 200, 150)}

    tracker.update({1: BoundingBox(140, 110, 160, 130)}, onibus, frame_idx=0)
    tracker.update({1: BoundingBox(500, 500, 520, 520)}, onibus, frame_idx=1)
    tracker.finalize(last_frame_idx=1)

    sessions = tracker.closed_sessions
    assert len(sessions) == 1
    assert sessions[0].duration_seconds(fps=30) < 1.0  # menos de 1s = passageiro, não trabalho


def test_switching_vehicle_without_leaving_closes_old_opens_new():
    tracker = VehicleProximityTracker()
    veiculos = {
        "1048": BoundingBox(0, 0, 50, 50),
        "1050": BoundingBox(200, 200, 250, 250),
    }
    tracker.update({1: BoundingBox(10, 10, 20, 20)}, veiculos, frame_idx=0)   # perto do 1048
    tracker.update({1: BoundingBox(215, 215, 225, 225)}, veiculos, frame_idx=1)  # pulou pro 1050
    tracker.finalize(last_frame_idx=1)

    sessions = sorted(tracker.closed_sessions, key=lambda s: s.start_frame)
    assert len(sessions) == 2
    assert sessions[0].vehicle_id == "1048"
    assert sessions[0].end_frame == 1
    assert sessions[1].vehicle_id == "1050"
    assert sessions[1].end_frame == 1


def test_total_seconds_per_vehicle_sums_across_people():
    tracker = VehicleProximityTracker()
    onibus = {"1048": BoundingBox(0, 0, 50, 50)}
    tracker.update({1: BoundingBox(10, 10, 20, 20), 2: BoundingBox(15, 15, 25, 25)}, onibus, frame_idx=0)
    tracker.update({1: BoundingBox(10, 10, 20, 20), 2: BoundingBox(15, 15, 25, 25)}, onibus, frame_idx=1)
    tracker.finalize(last_frame_idx=30)  # 30 frames

    totals = tracker.total_seconds_per_vehicle(fps=30)
    assert totals["1048"] == 2.0  # 1s de cada uma das 2 pessoas, somado
