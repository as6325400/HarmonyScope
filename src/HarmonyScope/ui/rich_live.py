from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.console import Group, Console
from HarmonyScope.core.constants import PITCH_CLASS_NAMES
from HarmonyScope.ui.table import make_pitch_class_table, make_detected_notes_table


class LiveMicUI:
    """Responsible for packaging the analysis results → Rich renderable"""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def build_renderable(
        self,
        chord,
        active_pcs,
        pitch_data_by_pc,
        detailed_peaks,
        segment_rms_db,
        total_voiced_frames,
        energy_thresh_db,
    ):
        renderables = []

        renderables.append(
            Text(
                f"Overall Segment Level: {segment_rms_db:.1f} dB │ "
                f"Threshold: {energy_thresh_db:.1f} dB │ "
                f"Voiced Frames: {total_voiced_frames}"
            )
        )
        renderables.append(make_pitch_class_table(pitch_data_by_pc))

        active_names = (
            ", ".join(PITCH_CLASS_NAMES[p] for p in sorted(active_pcs))
            or "[dim]None[/dim]"
        )
        renderables.append(Panel(active_names, title="Active PC Summary", expand=False))

        chord_text = f"[bold green]{chord}[/bold green]" if chord else "[dim]None[/dim]"
        renderables.append(Panel(chord_text, title="Chord Result", expand=False))

        return Group(*renderables)


def live_mic_loop(analyzer, ui: LiveMicUI, interval_sec: float):
    with Live(auto_refresh=False, screen=True) as live:
        for data in analyzer.stream_mic_live(interval_sec=interval_sec):
            # unpack the same tuple as the original
            (
                chord,
                active_pcs,
                pitch_data_by_pc,
                detailed_peaks,
                seg_rms_db,
                voiced_frames,
            ) = data
            live.update(
                ui.build_renderable(
                    chord,
                    active_pcs,
                    pitch_data_by_pc,
                    detailed_peaks,
                    seg_rms_db,
                    voiced_frames,
                    analyzer.frame_energy_thresh_db,
                ),
                refresh=True,
            )
