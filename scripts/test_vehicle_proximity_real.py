"""Script exploratório: roda YOLO detectando pessoa + veículo no mesmo vídeo,
alimenta o VehicleProximityTracker, e imprime o resultado bruto.

Não é o CLI oficial (dwell_report.py) — é a validação real pedida antes de
decidir se essa lógica entra na arquitetura principal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
from ultralytics import YOLO

from vehicle_proximity import BoundingBox, VehicleProximityTracker

PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

video_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/real_test_mechanic.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()

model = YOLO("yolov8n.pt")
results = model.track(
    source=video_path,
    classes=[PERSON_CLASS] + list(VEHICLE_CLASSES.keys()),
    conf=0.25,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    verbose=False,
)

tracker = VehicleProximityTracker(margin_ratio=0.5)
frame_idx = 0
pessoas_detectadas = set()
veiculos_detectados = {}  # id -> classe

for result in results:
    person_boxes = {}
    vehicle_boxes = {}
    if result.boxes is not None and result.boxes.id is not None:
        for box, track_id, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.id.tolist(), result.boxes.cls.tolist()
        ):
            x1, y1, x2, y2 = box
            tid = int(track_id)
            cls = int(cls)
            if cls == PERSON_CLASS:
                person_boxes[tid] = BoundingBox(x1, y1, x2, y2)
                pessoas_detectadas.add(tid)
            elif cls in VEHICLE_CLASSES:
                vehicle_boxes[tid] = BoundingBox(x1, y1, x2, y2)
                veiculos_detectados[tid] = VEHICLE_CLASSES[cls]

    tracker.update(person_boxes, vehicle_boxes, frame_idx)
    frame_idx += 1

tracker.finalize(last_frame_idx=frame_idx)

print(f"Vídeo: {video_path}")
print(f"Frames processados: {frame_idx} (fps={fps:.1f})")
print(f"IDs de pessoa detectados (brutos, antes de filtro): {sorted(pessoas_detectadas)}")
print(f"IDs de veículo detectados: {veiculos_detectados}")
print(f"Sessões de proximidade (pessoa perto de veículo): {len(tracker.closed_sessions)}")
for s in sorted(tracker.closed_sessions, key=lambda s: s.start_frame):
    dur = s.duration_seconds(fps)
    print(f"  pessoa {s.person_id} perto do veículo {s.vehicle_id}: {s.start_frame/fps:.1f}s -> {s.end_frame/fps:.1f}s ({dur:.1f}s)")

print("\nTotal por veículo:")
for vid, total in tracker.total_seconds_per_vehicle(fps).items():
    print(f"  veículo {vid}: {total:.1f}s")
