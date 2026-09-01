"""CLI: mede quanto tempo cada pessoa passa perto do veículo (zona definida
no frame), usando YOLOv8 (detecção) + ByteTrack (rastreamento por ID).

Exemplo:
    python src/dwell_report.py --source video.mp4 --zone 0.2,0.2,0.8,0.9 --save-video
"""

import argparse
import csv
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
from ultralytics import YOLO

from dwell_tracker import Zone, DwellTracker, filter_spurious_sessions
from fleet_id import parse_crop_arg, read_fleet_number, NAO_LIDO

PERSON_CLASS_ID = 0


def parse_zone_arg(zone_arg, width, height):
    x1, y1, x2, y2 = (float(v) for v in zone_arg.split(","))
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return Zone(x1, y1, x2, y2)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--model", default="yolov8n.pt", help="Pesos do YOLOv8 (baixa automático se não existir)")
    parser.add_argument(
        "--zone",
        default="0.2,0.2,0.8,0.9",
        help="Zona 'x1,y1,x2,y2'. Valores <=1 são tratados como razão do frame.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confiança mínima de detecção")
    parser.add_argument("--output", default="outputs/sessoes.csv", help="CSV de saída das sessões")
    parser.add_argument("--save-video", action="store_true", help="Salva vídeo anotado em outputs/annotated.mp4")
    parser.add_argument(
        "--min-session-seconds",
        type=float,
        default=1.0,
        help="Descarta sessões mais curtas que isso (detecção espúria por oclusão, não permanência real)",
    )
    parser.add_argument(
        "--fleet-crop",
        default=None,
        help=(
            "Liga a leitura do número de frota via OCR. 'x1,y1,x2,y2' com a região onde o "
            "número está pintado (valores <=1 viram razão do frame), ou 'auto' pro frame inteiro. "
            "Sem esta opção, nenhuma chamada de API é feita."
        ),
    )
    parser.add_argument(
        "--fleet-samples",
        type=int,
        default=3,
        help="Quantos frames amostrar pro OCR (1 chamada de API por amostra, o vídeo inteiro não é lido)",
    )
    return parser


def run(args):
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Vídeo não encontrado: {source_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"Não consegui abrir o vídeo: {source_path}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frota, concordaram, amostras = NAO_LIDO, 0, 0
    if args.fleet_crop:
        crop = parse_crop_arg(args.fleet_crop, width, height)
        frota, concordaram, amostras = read_fleet_number(
            source_path, total_frames, crop, num_samples=args.fleet_samples
        )

    zone = parse_zone_arg(args.zone, width, height)
    tracker = DwellTracker(zone=zone)

    writer = None
    if args.save_video:
        annotated_path = output_path.parent / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    model = YOLO(args.model)
    results = model.track(
        source=str(source_path),
        classes=[PERSON_CLASS_ID],
        conf=args.conf,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for result in results:
        tracked_objects = []
        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id in zip(result.boxes.xyxy.tolist(), result.boxes.id.tolist()):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tracked_objects.append((int(track_id), cx, cy))

        tracker.update(tracked_objects, frame_idx)

        if writer is not None:
            frame = result.plot()
            cv2.rectangle(
                frame, (int(zone.x1), int(zone.y1)), (int(zone.x2), int(zone.y2)), (0, 255, 255), 2
            )
            writer.write(frame)

        frame_idx += 1

    tracker.finalize(last_frame_idx=frame_idx)

    if writer is not None:
        writer.release()

    todas_sessoes = tracker.closed_sessions
    sessoes = filter_spurious_sessions(todas_sessoes, fps, min_seconds=args.min_session_seconds)
    descartadas = len(todas_sessoes) - len(sessoes)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frota", "track_id", "inicio_seg", "fim_seg", "duracao_seg"])
        writer_csv.writeheader()
        for session in sessoes:
            writer_csv.writerow({
                "frota": frota,
                "track_id": session.track_id,
                "inicio_seg": round(session.start_frame / fps, 1),
                "fim_seg": round(session.end_frame / fps, 1),
                "duracao_seg": round(session.duration_seconds(fps), 1),
            })

    totals = {}
    for session in sessoes:
        totals[session.track_id] = totals.get(session.track_id, 0.0) + session.duration_seconds(fps)

    if args.fleet_crop:
        if frota == NAO_LIDO:
            print("Número de frota: não consegui ler (o tempo abaixo continua válido)")
        else:
            print(f"Número de frota: {frota} ({concordaram} de {amostras} amostras concordaram)")
    print(f"Frames processados: {frame_idx}")
    print(f"Sessões registradas: {len(sessoes)} (descartadas {descartadas} espúrias < {args.min_session_seconds}s)")
    for track_id, total in totals.items():
        etiqueta = "" if frota == NAO_LIDO else f"VEÍCULO-{frota} · "
        print(f"  {etiqueta}ID {track_id}: {total:.1f}s no total perto do veículo")
    print(f"Log salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run(build_arg_parser().parse_args()))
