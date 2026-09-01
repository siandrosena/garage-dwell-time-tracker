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
    frota (3-4 dígitos). Retorna None se nada bater o padrão."""
    matches = FLEET_NUMBER_PATTERN.findall(raw_text)
    return matches[0] if matches else None


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
