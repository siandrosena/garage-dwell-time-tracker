"""Testes da camada de cola (fleet_id) entre o vídeo e o plate_ocr.

Regra de ouro do módulo: identificação é enriquecimento, não requisito — a
medição de tempo tem que sobreviver sem chave, sem internet e com OCR
errando. Todos os testes aqui usam mock; ZERO chamada real de rede/API.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fleet_id
from fleet_id import parse_crop_arg, read_fleet_number, NAO_LIDO
from plate_ocr import CropRegion


class FakeBuffer:
    """Substitui o retorno de cv2.imencode — só precisa saber virar bytes."""
    def tobytes(self):
        return b"fake-jpeg-bytes"


def _mock_video_layer(monkeypatch, frames=("f0", "f1", "f2")):
    """Remove a dependência de vídeo/imagem real: grab_frames devolve objetos
    quaisquer, crop_frame não faz nada com eles, e cv2.imencode nunca é
    chamado de verdade — só call_vision_ocr (mockado por teste) importa daqui pra frente.
    """
    monkeypatch.setattr(fleet_id, "grab_frames", lambda source_path, indices: list(frames))
    monkeypatch.setattr(fleet_id, "crop_frame", lambda frame, crop: frame)
    monkeypatch.setattr(fleet_id.cv2, "imencode", lambda ext, img: (True, FakeBuffer()))


# ---- parse_crop_arg ----

def test_parse_crop_arg_pixels_not_scaled():
    region = parse_crop_arg("100,50,300,200", width=640, height=480)
    assert region == CropRegion(100, 50, 300, 200)


def test_parse_crop_arg_ratio_scales_by_frame_size():
    region = parse_crop_arg("0.1,0.1,0.5,0.5", width=1000, height=800)
    assert region == CropRegion(100, 80, 500, 400)


def test_parse_crop_arg_auto_uses_full_frame():
    region = parse_crop_arg("auto", width=640, height=480)
    assert region == CropRegion(0, 0, 640, 480)


def test_parse_crop_arg_auto_is_case_and_space_insensitive():
    region = parse_crop_arg("  AUTO  ", width=640, height=480)
    assert region == CropRegion(0, 0, 640, 480)


# ---- read_fleet_number: degradação sem quebrar a medição de tempo ----

def test_no_api_key_returns_nao_lido_without_touching_video(monkeypatch):
    """Sem chave, nem vídeo nem API são tocados — zero custo, zero risco."""
    def _boom(*args, **kwargs):
        raise AssertionError("grab_frames não deveria ser chamado sem chave de API")

    monkeypatch.setattr(fleet_id, "grab_frames", _boom)
    monkeypatch.delenv(fleet_id.API_KEY_ENV, raising=False)

    numero, concordaram, amostras = read_fleet_number(
        "video.mp4", total_frames=300, crop=CropRegion(0, 0, 100, 100), api_key=None
    )
    assert (numero, concordaram, amostras) == (NAO_LIDO, 0, 0)


def test_no_frames_grabbed_returns_nao_lido(monkeypatch):
    """Vídeo não abre / não tem frame nenhum pra ler -> degrada, não quebra."""
    monkeypatch.setattr(fleet_id, "grab_frames", lambda source_path, indices: [])

    numero, concordaram, amostras = read_fleet_number(
        "video.mp4", total_frames=300, crop=CropRegion(0, 0, 100, 100), api_key="fake-key"
    )
    assert (numero, concordaram, amostras) == (NAO_LIDO, 0, 0)


def test_network_failure_on_every_sample_degrades_to_nao_lido(monkeypatch):
    """Rede caindo (timeout, DNS, o que for) em toda amostra não pode
    derrubar a medição — só zera a leitura de frota."""
    _mock_video_layer(monkeypatch)

    def _rede_caiu(image_bytes, api_key):
        raise ConnectionError("rede fora do ar")

    monkeypatch.setattr(fleet_id, "call_vision_ocr", _rede_caiu)

    numero, concordaram, amostras = read_fleet_number(
        "video.mp4", total_frames=300, crop=CropRegion(0, 0, 100, 100), num_samples=3, api_key="fake-key"
    )
    assert numero == NAO_LIDO
    assert concordaram == 0
    assert amostras == 3  # tentou as 3, todas falharam


def test_network_failure_on_some_samples_still_uses_the_rest(monkeypatch):
    """1 amostra falha na rede, as outras 2 concordam — não descarta tudo
    por causa de 1 erro isolado."""
    _mock_video_layer(monkeypatch)

    respostas = iter([ConnectionError("timeout"), "8256", "8256"])

    def _instavel(image_bytes, api_key):
        resposta = next(respostas)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    monkeypatch.setattr(fleet_id, "call_vision_ocr", _instavel)

    numero, concordaram, amostras = read_fleet_number(
        "video.mp4", total_frames=300, crop=CropRegion(0, 0, 100, 100), num_samples=3, api_key="fake-key"
    )
    assert numero == "8256"
    assert concordaram == 2
    assert amostras == 3


def test_ocr_with_no_text_degrades_to_nao_lido(monkeypatch):
    """OCR responde (sem erro), mas não achou nenhum número plausível de
    frota — mesma degradação segura de qualquer outra falha."""
    _mock_video_layer(monkeypatch)
    monkeypatch.setattr(fleet_id, "call_vision_ocr", lambda image_bytes, api_key: "garagem vazia, sem placa")

    numero, concordaram, amostras = read_fleet_number(
        "video.mp4", total_frames=300, crop=CropRegion(0, 0, 100, 100), num_samples=3, api_key="fake-key"
    )
    assert numero == NAO_LIDO
    assert concordaram == 0
    assert amostras == 3


def test_successful_majority_reading_end_to_end(monkeypatch):
    """Caminho feliz: 2 de 3 amostras concordam em '8256' -> vira a leitura final."""
    _mock_video_layer(monkeypatch)
    respostas = iter(["8256", "8Z56", "8256"])  # a 2a leitura "erra" 1 caractere (ruído de OCR)
    monkeypatch.setattr(fleet_id, "call_vision_ocr", lambda image_bytes, api_key: next(respostas))

    numero, concordaram, amostras = read_fleet_number(
        "video.mp4", total_frames=300, crop=CropRegion(0, 0, 100, 100), num_samples=3, api_key="fake-key"
    )
    assert numero == "8256"
    assert concordaram == 2
    assert amostras == 3
