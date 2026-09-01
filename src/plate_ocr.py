"""Lê o número de frota pintado no veículo via OCR (Google Cloud Vision),
recortando uma região fixa do frame — a câmera é estática, então o número
não muda de posição entre frames do mesmo vídeo.

Não roda OCR em todo frame de propósito: numa câmera fixa, o número pintado
no veículo não muda quadro a quadro, então algumas amostras (não centenas de
chamadas de API) já bastam pra uma leitura confiável — ver `sample_frames`.
"""

import base64
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

FLEET_NUMBER_PATTERN = re.compile(r"\b\d{3,4}\b")

# Camera de seguranca carimba data e hora no canto do frame. Sem remover isso,
# o OCR le o ANO ("2026") e devolve como se fosse o numero do veiculo — foi
# exatamente o que aconteceu no primeiro teste com video real.
DATA_HORA_PATTERN = re.compile(
    r"\d{4}[-/]\d{2}[-/]\d{2}"      # 2026-01-15
    r"|\d{2}[-/]\d{2}[-/]\d{4}"     # 01/09/2026
    r"|\d{1,2}:\d{2}(?::\d{2})?"    # 08:30:00
)

# Nenhuma frota e numerada como ano. Descartar essa faixa evita o falso
# positivo mais comum sem precisar recortar o frame.
FAIXA_ANO = range(1900, 2101)


@dataclass(frozen=True)
class CropRegion:
    """Região do frame onde o número aparece pintado (pixels)."""
    x1: int
    y1: int
    x2: int
    y2: int


def crop_frame(frame, region):
    return frame[region.y1:region.y2, region.x1:region.x2]


def call_vision_ocr(image_bytes, api_key):
    """Chama a Google Cloud Vision API (TEXT_DETECTION) com uma imagem crua
    (bytes já codificados, ex.: PNG/JPEG). Retorna o texto bruto detectado
    (string vazia se nada foi lido)."""
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }
    req = urllib.request.Request(
        f"{VISION_ENDPOINT}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    annotations = data.get("responses", [{}])[0].get("textAnnotations", [])
    if not annotations:
        return ""
    return annotations[0].get("description", "")


def extract_fleet_number(raw_text):
    """Do texto bruto do OCR, extrai o candidato mais plausível a número de
    frota (3-4 dígitos). Retorna None se nada bater o padrão.

    Ordem importa: primeiro apaga data/hora do carimbo da câmera, depois
    descarta o que parece ano, e só então prefere 4 dígitos sobre 3 — número
    de frota costuma ter 4, e sobra de outro texto costuma ter 3.
    """
    sem_data = DATA_HORA_PATTERN.sub(" ", raw_text)
    candidatos = [c for c in FLEET_NUMBER_PATTERN.findall(sem_data) if int(c) not in FAIXA_ANO]
    if not candidatos:
        return None
    de_quatro = [c for c in candidatos if len(c) == 4]
    return de_quatro[0] if de_quatro else candidatos[0]


def sample_frames(total_frames, num_samples=3):
    """Escolhe alguns índices de frame espalhados pelo vídeo (início, meio,
    fim) em vez de rodar OCR em cada frame — câmera fixa não precisa disso."""
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))
    step = total_frames // (num_samples + 1)
    return [step * (i + 1) for i in range(num_samples)]


def most_confident_reading(readings):
    """Dado várias leituras (algumas podem ser None), devolve a mais comum
    e quantas das amostras concordaram com ela."""
    valid = [r for r in readings if r]
    if not valid:
        return None, 0
    counts = Counter(valid)
    value, count = counts.most_common(1)[0]
    return value, count
