from dataclasses import dataclass, field, asdict


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    swing_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
