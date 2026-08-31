"""Mede tempo de permanência JUNTO DO VEÍCULO, não tempo "na área".

Pensa assim: um mecânico atravessando a garagem pra pegar uma peça não conta
como "trabalhando" — só conta o tempo em que ele está de verdade ao lado do
veículo. Isso é o que gera dado de negócio real: "o ônibus 1048 ficou 3h
parado, mas só teve mecânico junto dele por 40 min" é uma medida de GARGALO
DE PROCESSO, não de desempenho de funcionário.

Sobre isso ser vigilância ou não: o sistema não sabe quem é a pessoa, só que
"alguém" esteve perto de "um veículo" por X tempo. A pergunta que ele responde
é "quanto tempo esse veículo esperou por atendimento", nunca "essa pessoa
trabalhou o quanto" — ver README para a discussão completa de LGPD/consentimento.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    """Caixa retangular em pixels — mesma coisa que o YOLO devolve pra cada
    detecção (pessoa ou veículo)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def expanded(self, margin_ratio):
        """Cresce a caixa pra fora, proporcional ao próprio tamanho — um
        caminhão grande tem uma margem de 'perto' maior que um carro pequeno."""
        mx = self.width * margin_ratio
        my = self.height * margin_ratio
        return BoundingBox(self.x1 - mx, self.y1 - my, self.x2 + mx, self.y2 + my)

    def contains_point(self, point):
        px, py = point
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


def is_near_vehicle(person_box, vehicle_box, margin_ratio=0.5):
    """'Perto' = o centro da pessoa cai dentro da caixa do veículo, esticada
    pra fora em `margin_ratio` (0.5 = meia largura/altura do veículo de folga
    pra cada lado). Não usa distância em pixels fixa de propósito: a mesma
    distância em pixels significa coisas diferentes pra um carro pequeno
    (perto da câmera) e um caminhão grande (longe da câmera).
    """
    return vehicle_box.expanded(margin_ratio).contains_point((person_box.cx, person_box.cy))


def closest_vehicle(person_box, vehicles, margin_ratio=0.5):
    """vehicles: dict vehicle_id -> BoundingBox. Retorna o vehicle_id mais
    próximo dentre os que estão 'perto o suficiente', ou None se nenhum estiver.
    Usa distância entre centros só pra desempatar entre 2+ veículos próximos —
    o critério de PERTO já foi decidido por is_near_vehicle.
    """
    candidatos = [
        (vid, ((person_box.cx - box.cx) ** 2 + (person_box.cy - box.cy) ** 2) ** 0.5)
        for vid, box in vehicles.items()
        if is_near_vehicle(person_box, box, margin_ratio)
    ]
    if not candidatos:
        return None
    return min(candidatos, key=lambda item: item[1])[0]


@dataclass
class ProximitySession:
    person_id: int
    vehicle_id: object
    start_frame: int
    end_frame: int

    def duration_seconds(self, fps):
        return (self.end_frame - self.start_frame) / fps


@dataclass
class VehicleProximityTracker:
    """Mesma ideia do DwellTracker (Zone fixa), mas a 'zona' é um veículo que
    se move — e cada sessão fica associada a QUAL veículo, não só a 'dentro
    de um retângulo fixo do frame'.
    """
    margin_ratio: float = 0.5
    max_absence_frames: int = 45

    _open_sessions: dict = field(default_factory=dict)  # person_id -> (vehicle_id, start_frame)
    _last_seen: dict = field(default_factory=dict)  # person_id -> last_frame
    _closed_sessions: list = field(default_factory=list)

    def update(self, person_boxes, vehicle_boxes, frame_idx):
        """person_boxes: dict person_id -> BoundingBox (já com ID canônico
        resolvido — reaproveitar DwellTracker/_resolve_id pra isso antes de
        chamar aqui, a reconciliação de troca de ID não é responsabilidade
        desta classe).
        vehicle_boxes: dict vehicle_id -> BoundingBox.
        """
        for person_id, person_box in person_boxes.items():
            self._last_seen[person_id] = frame_idx
            near_id = closest_vehicle(person_box, vehicle_boxes, self.margin_ratio)
            open_entry = self._open_sessions.get(person_id)

            if near_id is not None and open_entry is None:
                self._open_sessions[person_id] = (near_id, frame_idx)
            elif near_id is not None and open_entry is not None and open_entry[0] != near_id:
                # trocou de veículo sem sair de "perto de algum" -> fecha a sessão do antigo, abre do novo
                old_vehicle_id, start_frame = open_entry
                self._closed_sessions.append(ProximitySession(person_id, old_vehicle_id, start_frame, frame_idx))
                self._open_sessions[person_id] = (near_id, frame_idx)
            elif near_id is None and open_entry is not None:
                vehicle_id, start_frame = self._open_sessions.pop(person_id)
                self._closed_sessions.append(ProximitySession(person_id, vehicle_id, start_frame, frame_idx))

        self._close_long_absent_sessions(frame_idx)

    def _close_long_absent_sessions(self, frame_idx):
        for person_id in list(self._open_sessions.keys()):
            last_frame = self._last_seen.get(person_id, frame_idx)
            if frame_idx - last_frame > self.max_absence_frames:
                vehicle_id, start_frame = self._open_sessions.pop(person_id)
                self._closed_sessions.append(ProximitySession(person_id, vehicle_id, start_frame, last_frame))

    def finalize(self, last_frame_idx):
        for person_id, (vehicle_id, start_frame) in list(self._open_sessions.items()):
            self._closed_sessions.append(ProximitySession(person_id, vehicle_id, start_frame, last_frame_idx))
        self._open_sessions.clear()

    @property
    def closed_sessions(self):
        return list(self._closed_sessions)

    def total_seconds_per_vehicle(self, fps):
        totals = {}
        for session in self._closed_sessions:
            totals[session.vehicle_id] = totals.get(session.vehicle_id, 0.0) + session.duration_seconds(fps)
        return totals

    def total_seconds_per_person(self, fps):
        totals = {}
        for session in self._closed_sessions:
            totals[session.person_id] = totals.get(session.person_id, 0.0) + session.duration_seconds(fps)
        return totals
