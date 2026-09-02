import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plate_ocr import extract_fleet_number, sample_frames, most_confident_reading


def test_extract_fleet_number_from_clean_text():
    assert extract_fleet_number("8256") == "8256"


def test_extract_fleet_number_ignores_surrounding_noise():
    assert extract_fleet_number("FROTA\n8256\nBRT") == "8256"


def test_extract_fleet_number_returns_none_when_no_digits():
    assert extract_fleet_number("sem numero nenhum aqui") is None


def test_extract_fleet_number_ignores_short_numbers_like_years_digits():
    # numero de 1-2 digitos (ex: "20" de "2026") nao deveria contar como frota
    assert extract_fleet_number("20") is None
    assert extract_fleet_number("8256") == "8256"


def test_sample_frames_spreads_across_video_not_every_frame():
    samples = sample_frames(total_frames=3000, num_samples=3)
    assert len(samples) == 3
    assert samples == sorted(samples)
    assert all(0 <= s < 3000 for s in samples)
    # nao pode ser so os 3 primeiros frames (isso nao "espalharia" no video)
    assert samples[-1] > 1000


def test_sample_frames_handles_short_video():
    samples = sample_frames(total_frames=2, num_samples=3)
    assert samples == [0, 1]


def test_sample_frames_handles_empty_video():
    assert sample_frames(total_frames=0, num_samples=3) == []


def test_most_confident_reading_picks_majority():
    value, count = most_confident_reading(["8256", "8256", "8Z56", None])
    assert value == "8256"
    assert count == 2


def test_most_confident_reading_all_none_returns_none():
    value, count = most_confident_reading([None, None])
    assert value is None
    assert count == 0


def test_extract_fleet_number_real_capture_ignores_camera_timestamp():
    """Falha real capturada em campo (não hipótese): a câmera de segurança
    carimba data/hora no frame, e o OCR devolvia o ano do carimbo (ex.: 2026)
    como se fosse o número da frota. Estrutura fiel ao texto bruto real
    devolvido pela Vision API (carimbo + código de rota + número de frota +
    marca-d'água do fabricante da câmera), com valores trocados por fictícios."""
    texto_capturado = "2026-01-15 08:30:00)\nRT-014\n8256\nCamMark"
    assert extract_fleet_number(texto_capturado) == "8256"


def test_extract_fleet_number_ignores_br_date_format():
    assert extract_fleet_number("01/09/2026\n8256") == "8256"


def test_extract_fleet_number_ignores_time_without_seconds():
    assert extract_fleet_number("10:27\n8256") == "8256"


def test_extract_fleet_number_returns_three_digits_when_thats_all_there_is():
    assert extract_fleet_number("frota 041 chegou") == "041"


def test_extract_fleet_number_with_two_four_digit_numbers_picks_first():
    """Ambiguidade real: 2 números de 4 dígitos no mesmo texto (ex.: número
    da frota E um código de peça/ordem de serviço). Não tem como saber qual
    é o certo só pelo texto — a escolha aqui é 'o primeiro que aparece',
    documentada como escolha, não como certeza. Recortar a região certa do
    frame (crop) evita esse caso na prática; ver README."""
    assert extract_fleet_number("8256 os-9382") == "8256"
