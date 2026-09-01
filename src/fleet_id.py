"""Cola entre o vídeo e o `plate_ocr`: pega algumas amostras de frame,
recorta a região onde o número de frota aparece pintado e devolve uma leitura
única, com quantas amostras concordaram.

Fica separado do `dwell_report` de propósito: a medição de tempo tem que
continuar funcionando mesmo sem chave de API, sem internet ou com o OCR
errando. Identificação é enriquecimento, não requisito.
"""

import os
import sys

import cv2

from plate_ocr import (
    CropRegion,
    crop_frame,
    call_vision_ocr,
    extract_fleet_number,
    sample_frames,
    most_confident_reading,
)

API_KEY_ENV = "GOOGLE_VISION_API_KEY"

NAO_LIDO = "NAO_LIDO"


def parse_crop_arg(crop_arg, width, height):
    """Aceita 'x1,y1,x2,y2' (pixels, ou razão do frame se todos <= 1) ou a
    palavra 'auto', que usa o frame inteiro."""
    if crop_arg.strip().lower() == "auto":
        return CropRegion(0, 0, width, height)
    x1, y1, x2, y2 = (float(v) for v in crop_arg.split(","))
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return CropRegion(int(x1), int(y1), int(x2), int(y2))


def grab_frames(source_path, indices):
    """Lê só os frames pedidos, sem varrer o vídeo inteiro."""
    frames = []
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return frames
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def read_fleet_number(source_path, total_frames, crop, num_samples=3, api_key=None):
    """Devolve (numero, amostras_que_concordaram, total_de_amostras_lidas).

    Nunca levanta exceção: qualquer falha (sem chave, rede fora, cota estourada,
    OCR sem texto) vira NAO_LIDO, e a medição de tempo segue normal.
    """
    api_key = api_key or os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        print(
            f"OCR desligado: variável {API_KEY_ENV} não está definida — "
            "o tempo é medido normalmente, só sem o número do veículo.",
            file=sys.stderr,
        )
        return NAO_LIDO, 0, 0

    indices = sample_frames(total_frames, num_samples=num_samples)
    frames = grab_frames(source_path, indices)
    if not frames:
        return NAO_LIDO, 0, 0

    leituras = []
    for frame in frames:
        recorte = crop_frame(frame, crop)
        ok, buffer = cv2.imencode(".jpg", recorte)
        if not ok:
            leituras.append(None)
            continue
        try:
            texto = call_vision_ocr(buffer.tobytes(), api_key)
        except Exception as exc:  # rede, cota, chave inválida — não pode derrubar a medição
            print(f"OCR falhou numa amostra ({type(exc).__name__}): seguindo sem ela.", file=sys.stderr)
            leituras.append(None)
            continue
        leituras.append(extract_fleet_number(texto))

    numero, concordaram = most_confident_reading(leituras)
    return (numero or NAO_LIDO), concordaram, len(leituras)
