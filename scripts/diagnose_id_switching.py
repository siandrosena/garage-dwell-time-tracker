"""Script exploratório: confirma (ou refuta) se IDs específicos do ByteTrack
sao trocas da MESMA pessoa, antes de qualquer correcao.

Prova exigida: os IDs suspeitos NUNCA aparecem no mesmo frame, e ocupam
posicoes proximas quando um termina e o outro comeca.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultralytics import YOLO

PERSON_CLASS_ID = 0

video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/real_test_mechanic.mp4"
tracker_yaml = sys.argv[2] if len(sys.argv) > 2 else "bytetrack.yaml"
ids_suspeitos = set(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else None

model = YOLO("yolov8n.pt")
inicio = time.time()
results = model.track(
    source=video_path,
    classes=[PERSON_CLASS_ID],
    conf=0.35,
    tracker=tracker_yaml,
    persist=True,
    stream=True,
    verbose=False,
)

# track_id -> lista de (frame_idx, cx, cy)
historico = {}
presenca_por_frame = {}  # frame_idx -> {track_id: (cx,cy)}

frame_idx = 0
for result in results:
    presentes = {}
    if result.boxes is not None and result.boxes.id is not None:
        for box, tid in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            tid = int(tid)
            historico.setdefault(tid, []).append((frame_idx, cx, cy))
            presentes[tid] = (cx, cy)
    presenca_por_frame[frame_idx] = presentes
    frame_idx += 1

total_frames = frame_idx
duracao = time.time() - inicio
print(f"Vídeo: {video_path} | tracker: {tracker_yaml} | {total_frames} frames | {len(historico)} IDs únicos | {duracao:.1f}s ({total_frames/duracao:.1f} fps processados)\n")

# resumo geral: primeiro/ultimo frame de cada ID
resumo = []
for tid, pontos in historico.items():
    primeiro = pontos[0][0]
    ultimo = pontos[-1][0]
    cx_medio = sum(p[1] for p in pontos) / len(pontos)
    cy_medio = sum(p[2] for p in pontos) / len(pontos)
    resumo.append((tid, primeiro, ultimo, len(pontos), cx_medio, cy_medio))
resumo.sort(key=lambda r: r[1])

print(f"{'ID':>4} {'1o frame':>9} {'ult frame':>10} {'n frames':>9} {'cx medio':>9} {'cy medio':>9}")
for tid, primeiro, ultimo, n, cx, cy in resumo:
    marca = " <--- suspeito" if ids_suspeitos and tid in ids_suspeitos else ""
    print(f"{tid:>4} {primeiro:>9} {ultimo:>10} {n:>9} {cx:>9.1f} {cy:>9.1f}{marca}")

if ids_suspeitos:
    print(f"\n--- Verificação dos IDs suspeitos {sorted(ids_suspeitos)} ---")
    # 1) nunca aparecem juntos no mesmo frame?
    conflito = False
    for f_idx, presentes in presenca_por_frame.items():
        presentes_suspeitos = [tid for tid in presentes if tid in ids_suspeitos]
        if len(presentes_suspeitos) > 1:
            conflito = True
            print(f"  CONFLITO no frame {f_idx}: {presentes_suspeitos} aparecem JUNTOS (não pode ser troca de ID)")
    if not conflito:
        print("  OK: nenhum dos IDs suspeitos aparece no mesmo frame que outro -> consistente com TROCA DE ID (mesma pessoa)")

    # 2) gap temporal e distância espacial entre o fim de um e o início do próximo
    suspeitos_ordenados = sorted(
        [(tid, historico[tid][0][0], historico[tid][-1][0], historico[tid][-1][1], historico[tid][-1][2], historico[tid][0][1], historico[tid][0][2])
         for tid in ids_suspeitos if tid in historico],
        key=lambda r: r[1],
    )
    print("\n  Transições (fim de um ID -> início do próximo):")
    for i in range(len(suspeitos_ordenados) - 1):
        tid_a, ini_a, fim_a, cx_fim_a, cy_fim_a, cx_ini_a, cy_ini_a = suspeitos_ordenados[i]
        tid_b, ini_b, fim_b, cx_fim_b, cy_fim_b, cx_ini_b, cy_ini_b = suspeitos_ordenados[i + 1]
        gap_frames = ini_b - fim_a
        dist = ((cx_ini_b - cx_fim_a) ** 2 + (cy_ini_b - cy_fim_a) ** 2) ** 0.5
        print(f"  ID {tid_a} termina no frame {fim_a} em ({cx_fim_a:.0f},{cy_fim_a:.0f}) -> ID {tid_b} começa no frame {ini_b} em ({cx_ini_b:.0f},{cy_ini_b:.0f}) | gap={gap_frames} frames | distância={dist:.1f}px")
