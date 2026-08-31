"""Mede quanto tempo cada pessoa rastreada passa dentro de uma zona
(ex.: "perto do veículo em manutenção"), a partir de centroides por frame.

Reaproveita o mesmo princípio de reconciliação de ID por troca do projeto
contador-onibus (crossing.py): o rastreador de vídeo perde e reatribui IDs
o tempo todo, e sem tratar isso o tempo de permanência de uma mesma pessoa
vira várias sessões picadas em vez de uma sessão contínua.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Zone:
    """Zona retangular no frame (pixels): (x1, y1) canto superior-esquerdo,
    (x2, y2) canto inferior-direito."""
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, point):
        px, py = point
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


@dataclass
class DwellSession:
    track_id: int
    start_frame: int
    end_frame: int

    def duration_seconds(self, fps):
        return (self.end_frame - self.start_frame) / fps


@dataclass
class DwellTracker:
    zone: Zone
    max_reassign_dist: float = 60.0
    max_reassign_gap: int = 15
    max_absence_frames: int = 45  # sessão fecha sozinha se a pessoa some por mais que isso

    _last_seen: dict = field(default_factory=dict)  # canonical_id -> (cx, cy, frame_idx)
    _open_sessions: dict = field(default_factory=dict)  # canonical_id -> start_frame
    _lost_tracks: dict = field(default_factory=dict)  # raw/canonical id -> (cx, cy, frame_idx)
    _id_aliases: dict = field(default_factory=dict)  # raw tracker id -> canonical id
    _closed_sessions: list = field(default_factory=list)

    def _resolve_id(self, raw_id, centroid, frame_idx):
        if raw_id in self._id_aliases:
            return self._id_aliases[raw_id]
        if raw_id in self._last_seen:
            return raw_id

        best_match, best_dist = None, None
        for lost_id, (lx, ly, lframe) in self._lost_tracks.items():
            if frame_idx - lframe > self.max_reassign_gap:
                continue
            dist = ((centroid[0] - lx) ** 2 + (centroid[1] - ly) ** 2) ** 0.5
            if dist <= self.max_reassign_dist and (best_dist is None or dist < best_dist):
                best_match, best_dist = lost_id, dist

        if best_match is not None:
            self._id_aliases[raw_id] = best_match
            del self._lost_tracks[best_match]
            return best_match

        return raw_id

    def update(self, tracked_objects, frame_idx):
        """tracked_objects: iterável de (raw_id, cx, cy). Retorna nada;
        consultar closed_sessions ao final (via finalize) pra ver o resultado.
        """
        seen_this_frame = set()

        for raw_id, cx, cy in tracked_objects:
            canonical_id = self._resolve_id(raw_id, (cx, cy), frame_idx)
            seen_this_frame.add(canonical_id)
            self._last_seen[canonical_id] = (cx, cy, frame_idx)

            inside = self.zone.contains((cx, cy))
            was_open = canonical_id in self._open_sessions

            if inside and not was_open:
                self._open_sessions[canonical_id] = frame_idx
            elif not inside and was_open:
                start_frame = self._open_sessions.pop(canonical_id)
                self._closed_sessions.append(DwellSession(canonical_id, start_frame, frame_idx))

        for tracked_id in list(self._last_seen.keys()):
            if tracked_id not in seen_this_frame:
                cx, cy, last_frame = self._last_seen[tracked_id]
                if frame_idx - last_frame == 1:
                    self._lost_tracks[tracked_id] = (cx, cy, last_frame)

        self._close_long_absent_sessions(frame_idx)

    def _close_long_absent_sessions(self, frame_idx):
        """Sem isso, uma pessoa que o rastreador perde de vez no meio do vídeo
        (sai de cena, fica oculta demais pra sempre) mantém a sessão "aberta"
        até o finalize() do vídeo inteiro — inflando a duração registrada bem
        além do tempo real em que ela foi vista de verdade."""
        for canonical_id in list(self._open_sessions.keys()):
            _, _, last_frame = self._last_seen[canonical_id]
            if frame_idx - last_frame > self.max_absence_frames:
                start_frame = self._open_sessions.pop(canonical_id)
                self._closed_sessions.append(DwellSession(canonical_id, start_frame, last_frame))

    def finalize(self, last_frame_idx):
        """Fecha sessões que ainda estavam abertas quando o vídeo acabou."""
        for canonical_id, start_frame in list(self._open_sessions.items()):
            self._closed_sessions.append(DwellSession(canonical_id, start_frame, last_frame_idx))
        self._open_sessions.clear()

    @property
    def closed_sessions(self):
        return list(self._closed_sessions)

    def total_seconds_per_id(self, fps):
        totals = {}
        for session in self._closed_sessions:
            totals[session.track_id] = totals.get(session.track_id, 0.0) + session.duration_seconds(fps)
        return totals


def filter_spurious_sessions(sessions, fps, min_seconds=1.0):
    """Descarta sessões curtas demais pra serem uma permanência real.

    Achado em vídeo real: durante oclusão parcial (ex.: mecânico agachado
    embaixo do carro), o YOLO às vezes enxerga a mesma pessoa como uma 2ª
    caixa por poucos frames — isso vira uma sessão "fantasma" de menos de
    1 segundo, de gente que nunca esteve ali de verdade. Uma permanência
    real de trabalho dura segundos/minutos, não frações de segundo.
    """
    return [s for s in sessions if s.duration_seconds(fps) >= min_seconds]
