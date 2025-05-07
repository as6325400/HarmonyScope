from typing import Set, Tuple, Optional
from .constants import PITCH_CLASS_NAMES, CHORD_RELATIONS

def identify_chord(pitches: Set[int]) -> Optional[str]:
    if not pitches:
        return None
    best: Optional[Tuple[int,str]] = None   # (complexity, name)
    for root_pc in pitches:
        intervals = {(pc - root_pc) % 12 for pc in pitches}
        intervals.add(0)
        for suffix, rel in CHORD_RELATIONS:
            if intervals.issubset(rel):
                cmpx = len(rel)
                name = f"{PITCH_CLASS_NAMES[root_pc]}{suffix}"
                if best is None or cmpx < best[0]:
                    best = (cmpx, name)
                break
    return best[1] if best else None
