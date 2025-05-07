from HarmonyScope.io.mic_reader import list_input_devices, MicReader
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
import argparse, sys
import questionary
from questionary import Choice
from HarmonyScope import set_verbosity
from rich.live import Live
from rich.table import Table

def choose_device_interactive() -> int:
    """Arrow‑key selector – returns the chosen PortAudio device id."""
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No input devices found")

    # Map to questionary Choice objects
    choices = [
        Choice(title=f"[{idx}] {name}", value=idx) for idx, name in devices
    ]

    device_id = questionary.select(
        "Select input device (arrow keys, <Enter> to confirm):",
        choices=choices,
        qmark="❯",
        pointer="▶",
        instruction="",
    ).ask()

    if device_id is None:                # <Esc> or Ctrl‑C
        raise KeyboardInterrupt
    return device_id

def make_table(pitch_data):
    """
    pitch_data: a list of dicts, each dict has:
      {'index', 'name', 'energy_db', 'delta_db', 'active'}
    """
    table = Table(title="Active-pitch debug dump")
    table.add_column("Note", justify="center")
    table.add_column("Level (dB)", justify="center")
    table.add_column("Delta", justify="center")
    table.add_column("Active", justify="center")

    for pitch in pitch_data:
        level = pitch['energy_db']
        delta = pitch['delta_db']
        active = "✔" if pitch['active'] else ""
        table.add_row(
            pitch['name'],
            f"{level:+.1f} dB",
            f"{delta:>3.1f}",
            active
        )
    return table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int,
                    help="device id (use --device -1 to list & choose interactively)")
    args = ap.parse_args()
    
    dev_id = args.device if args.device is not None else choose_device_interactive()
    reader = MicReader(device=dev_id)
    ana    = ChordAnalyzer(reader=reader)
    
    with Live(auto_refresh=False) as live:
        for chord, pitch_data in ana.stream_mic_live():    # pitch_data = list of dicts
            table_render = make_table(pitch_data)
            # 把 chord 也加入 table 最下面一行
            table_render.add_row("", "", "", f"[bold green]Live chord: {chord}[/bold green]")
            
            live.update(table_render, refresh=True)