from typing import Set, Tuple, Optional, List, Dict, Any
from .constants import PITCH_CLASS_NAMES, CHORD_RELATIONS
import numpy as np  # Import numpy for isnan check


def identify_chord(
    active_pitch_classes: Set[int], detailed_peak_detections: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Identifies a chord from a set of active pitch classes (0-11),
    informed by the specific detected notes (with octaves) to help determine the root.
    Appends bass note notation (e.g., C/E) if the lowest detected note differs
    from the identified chord root.

    Args:
        active_pitch_classes (Set[int]): A set of active pitch class indices (0-11),
                                         determined by persistence across frames.
        detailed_peak_detections (List[Dict[str, Any]]): A list of dictionaries, each
                                                        representing a single detected spectral
                                                        peak (note) including its MIDI note,
                                                        pitch class, and octave.

    Returns:
        Optional[str]: The identified chord name (e.g., "C", "Am", "G7", "C/E") or None if no chord is identified.
    """
    if not active_pitch_classes:
        # If no pitch classes are considered 'active' based on frame ratio, no chord can be identified.
        return None

    # 1. Determine the bass note (lowest MIDI note) from all detected peaks
    # This is the actual lowest sounding note, regardless of whether its PC
    # was deemed "active" by the frame count threshold.
    lowest_midi_note = None
    actual_bass_pc = None  # Pitch class of the absolute lowest detected note

    # Filter out potential invalid MIDI notes (-1 or NaN) before finding the minimum
    valid_detections = [
        d
        for d in detailed_peak_detections
        if d.get("midi_note") is not None
        and not np.isnan(d.get("midi_note"))
        and d.get("midi_note", -1) >= 0
    ]

    if valid_detections:
        # Find the detection with the minimum valid MIDI note value
        lowest_detection = min(valid_detections, key=lambda x: x["midi_note"])
        lowest_midi_note = lowest_detection["midi_note"]
        actual_bass_pc = (
            lowest_midi_note % 12
        )  # Calculate the pitch class of the bass note

    # 2. Create a list of candidate root pitch classes for chord identification
    # We prioritize the pitch class of the actual bass note *if* it's also
    # considered "active" (i.e., present in the active_pitch_classes set).
    candidate_roots = list(active_pitch_classes)  # Start with all active PCs

    # If we found a bass note AND its pitch class is among the active ones,
    # put it first in the list of candidates to check as the root.
    if actual_bass_pc is not None and actual_bass_pc in active_pitch_classes:
        # Move the bass PC to the front of the list to check it first
        candidate_roots.insert(
            0, candidate_roots.pop(candidate_roots.index(actual_bass_pc))
        )

    # Ensure unique candidates while keeping the prioritized order
    seen = set()
    ordered_candidate_roots = []
    for pc in candidate_roots:
        if pc not in seen:
            seen.add(pc)
            ordered_candidate_roots.append(pc)

    # If even after prioritizing the bass note, there are no candidate roots
    # derived from the active pitch classes (e.g., active_pitch_classes was empty,
    # which is caught at the beginning, or the bass PC wasn't active and no other
    # PCs were active - unlikely but possible), we can't proceed.
    if not ordered_candidate_roots:
        return None

    best: Optional[Tuple[int, str, int]] = (
        None  # (complexity, chord_name, identified_root_pc)
    )

    # 3. Iterate through candidate roots and check chord patterns using active pitch classes
    for root_pc_candidate in ordered_candidate_roots:
        # Calculate the pitch class intervals relative to the current candidate root PC
        # We use the *set of active pitch classes* for interval checking, as this
        # is what defines the chord type in standard theory.
        intervals = {(pc - root_pc_candidate) % 12 for pc in active_pitch_classes}
        # The root itself (interval 0) must be included for pattern matching
        intervals.add(0)

        # Check against predefined chord patterns, sorted roughly by complexity (smaller first)
        for suffix, rel in CHORD_RELATIONS:
            # If the set of active pitch class intervals is a subset of the required
            # pitch classes for this chord pattern relative to the root candidate PC...
            cmpx = (
                len(rel ^ intervals),
                len(rel),
            )  # complexity = size of rel + size of difference
            current_chord_name = f"{PITCH_CLASS_NAMES[root_pc_candidate]}{suffix}"

            # Store the best match found so far. Prefer less complex patterns.
            if best is None or cmpx < best[0]:
                # Store complexity, chord name based on candidate root, and the candidate root PC itself
                best = (cmpx, current_chord_name, root_pc_candidate)
            # No break here. We check all candidate roots and patterns to find the overall simplest chord
            # that matches the active pitch classes from any of the candidate roots.
            # If we broke here, we'd only consider the first pattern match for the current root candidate.
            # By not breaking, we might find a simpler chord definition for a later root candidate.

    # 4. Determine the final chord name, adding bass note notation if needed
    if best is not None:
        # A chord pattern was successfully matched based on the active pitch classes
        identified_root_pc = best[
            2
        ]  # This is the root PC of the chord identified by pattern matching

        # Check if we found a bass note AND its pitch class is different from the identified chord root PC
        if actual_bass_pc is not None and actual_bass_pc != identified_root_pc:
            # The actual lowest note is different from the identified chord's root.
            # Add the slash notation for the bass note.
            bass_note_name = PITCH_CLASS_NAMES[actual_bass_pc]
            return f"{best[1]}/{bass_note_name}"
        else:
            # The actual bass note matches the identified root PC, or no bass note was detected.
            # Return the identified chord name without slash notation.
            return best[1]
    else:
        # No chord was identified at all based on the active pitch classes
        return None
