"""Script exploratório (não faz parte do pipeline): mede em que confiança
(se em alguma) o YOLO padrão reconhece o carro erguido no elevador como
QUALQUER classe de veículo do COCO, e o que ele confunde com o quê.
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultralytics import YOLO

COCO_NAMES = {
    2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    56: "chair", 60: "dining table", 62: "tv", 13: "bench", 0: "person",
}

video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/real_test_mechanic.mp4"

model = YOLO("yolov8n.pt")

# conf bem baixo (0.01) e SEM filtro de classe: quero ver tudo que o modelo
# cogitou, mesmo com pouquissima confianca, pra saber se o carro aparece
# como QUALQUER coisa reconhecivel.
results = model.predict(
    source=video_path,
    conf=0.01,
    iou=0.5,
    stream=True,
    verbose=False,
)

max_conf_por_classe = defaultdict(float)
frames_com_deteccao_por_classe = defaultdict(int)
total_frames = 0

for result in results:
    total_frames += 1
    if result.boxes is None:
        continue
    for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        cls = int(cls)
        if conf > max_conf_por_classe[cls]:
            max_conf_por_classe[cls] = conf
        frames_com_deteccao_por_classe[cls] += 1

print(f"Vídeo: {video_path} — {total_frames} frames analisados (conf mínima 0.01, sem filtro de classe)\n")
print("Classes de VEÍCULO (COCO) — confiança máxima vista em qualquer frame:")
for cls_id, nome in [(2, "car"), (3, "motorcycle"), (5, "bus"), (7, "truck")]:
    max_conf = max_conf_por_classe.get(cls_id, 0.0)
    n_frames = frames_com_deteccao_por_classe.get(cls_id, 0)
    print(f"  {nome:12s} (id {cls_id}): confiança máx = {max_conf:.3f} | apareceu em {n_frames}/{total_frames} frames")

print("\nTop 10 classes (qualquer tipo) por confiança máxima vista — pra saber o que o modelo enxergou no lugar do carro:")
model_names = model.names
ranking = sorted(max_conf_por_classe.items(), key=lambda kv: -kv[1])[:10]
for cls_id, conf in ranking:
    nome = model_names.get(cls_id, f"id{cls_id}")
    n_frames = frames_com_deteccao_por_classe.get(cls_id, 0)
    print(f"  {nome:20s}: confiança máx = {conf:.3f} | {n_frames}/{total_frames} frames")
