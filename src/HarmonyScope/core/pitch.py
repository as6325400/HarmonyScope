import numpy as np
import librosa
import logging
from .constants import PITCH_CLASS_NAMES

logger = logging.getLogger(__name__)

def active_pitches_array(y, sr, *,
                         frame_energy_thresh_db=-40,
                         delta_db=2):
    """Return a set of active pitch‑class indices for a waveform array."""
    # 1. frame RMS
    rms = librosa.feature.rms(y=y)[0]
    voiced_frames = rms > librosa.db_to_amplitude(frame_energy_thresh_db, ref=np.max(rms))
    if voiced_frames.sum() == 0:
        # logger.debug("No voiced frames detected")
        return set()

    # For each pitch‑class row, take its maximum energy; any row whose value is within Δ dB of the peak is marked as active.
    # 2. chroma on voiced frames
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)[:, voiced_frames]
    pc_energy = librosa.amplitude_to_db(chroma, ref=np.max).max(axis=1)

    # 3. delta‑window rule
    peak = pc_energy.max()
    act_idx = np.where(pc_energy >= peak - delta_db)[0]
    active = set(act_idx)
    
    table = []
    for i, name in enumerate(PITCH_CLASS_NAMES):
        diff = peak - pc_energy[i]
        info = {
            'index': i,
            'name': name,
            'energy_db': pc_energy[i],
            'delta_db': diff,
            'active': (i in active)
        }
        table.append(info)
        
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Active‑pitch debug dump:")
        for i, name in enumerate(PITCH_CLASS_NAMES):
            diff = peak - pc_energy[i]
            flag = "✔" if i in active else " "
            logger.debug(
                f"{name:2}: {pc_energy[i]:6.1f} dB  "
                f"(Δ={diff:4.1f}) {flag}"
            )
            
    return active, table
