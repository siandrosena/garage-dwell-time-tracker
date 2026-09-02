"""Compara configuracoes de tracker (bytetrack com buffers diferentes, e
botsort com ReID) no mesmo video real, medindo:
- total de IDs unicos de pessoa
- quantos IDs distintos cobrem a "regiao do mecanico" (onde 4/16/25 apareceram
  na config padrao) -> se cair pra perto de 1, a troca de ID foi corrigida
- maior "espalhamento" espacial de um unico ID (cx max - cx min) -> se ficar
  grande demais, e sinal de ter juntado 2 PESSOAS DIFERENTES no mesmo ID
  (o erro oposto, buffer grande demais)
- tempo de processamento
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultralytics import YOLO

PERSON_CLASS_ID = 0
REGIAO_MECANICO = (40, 250, 110, 350)  # x1,y1,x2,y2 - onde 4/16/25 apareceram na config padrao

video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/real_test_mechanic.mp4"
configs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bytetrack.yaml"]


def roda(tracker_yaml):
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
    historico = {}
    frame_idx = 0
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            for box, tid in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                historico.setdefault(int(tid), []).append((frame_idx, cx, cy))
        frame_idx += 1
    duracao = time.time() - inicio

    rx1, ry1, rx2, ry2 = REGIAO_MECANICO
    ids_na_regiao = []
    for tid, pontos in historico.items():
        n_na_regiao = sum(1 for _, cx, cy in pontos if rx1 <= cx <= rx2 and ry1 <= cy <= ry2)
        if n_na_regiao >= 5:
            ids_na_regiao.append(tid)

    maior_espalhamento = 0.0
    id_mais_espalhado = None
    for tid, pontos in historico.items():
        cxs = [p[1] for p in pontos]
        espalhamento = max(cxs) - min(cxs)
        if espalhamento > maior_espalhamento:
            maior_espalhamento = espalhamento
            id_mais_espalhado = tid

    return {
        "tracker": tracker_yaml,
        "frames": frame_idx,
        "ids_totais": len(historico),
        "ids_regiao_mecanico": sorted(ids_na_regiao),
        "n_ids_regiao_mecanico": len(ids_na_regiao),
        "maior_espalhamento_cx": maior_espalhamento,
        "id_mais_espalhado": id_mais_espalhado,
        "tempo_s": duracao,
    }


resultados = []
for cfg in configs:
    print(f"Rodando {cfg}...", flush=True)
    r = roda(cfg)
    resultados.append(r)

print(f"\nVídeo: {video_path}\n")
print(f"{'tracker':<28} {'IDs totais':>10} {'IDs regiao mec.':>16} {'espalh. max (px)':>17} {'ID espalhado':>13} {'tempo (s)':>10}")
for r in resultados:
    print(f"{r['tracker']:<28} {r['ids_totais']:>10} {r['n_ids_regiao_mecanico']:>16} {r['maior_espalhamento_cx']:>17.1f} {str(r['id_mais_espalhado']):>13} {r['tempo_s']:>10.1f}")

print("\nDetalhe dos IDs na região do mecânico por config:")
for r in resultados:
    print(f"  {r['tracker']}: {r['ids_regiao_mecanico']}")
