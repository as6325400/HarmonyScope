from HarmonyScope.ui.plot import plot_chroma, plot_spec, plot_wave
import gradio as gr, pandas as pd
from typing import List, Dict, Any

Frame = Dict[str, Any]


def build_gradio_app(frames: List[Frame], hop: float, win: float) -> gr.Blocks:
    """Minimal-whitespace responsive layout using only props compatible with older Gradio versions."""

    def render(idx: int):
        f = frames[idx]
        w_fig = plot_wave(f["wave"], f["t"], win)
        s_fig = plot_spec(f["spec"], f["t"], win)
        c_fig = plot_chroma(f["chroma"], f["t"], win)
        t_range = f"{f['t']:.2f} – {(f['t']+win):.2f} s"
        df = pd.DataFrame(f["pc_summary"])[["name", "detection_count"]]
        df.columns = ["PC", "Frames"]
        return w_fig, s_fig, c_fig, f["chord"], t_range, df

    # Styles: shrink container + limit table height via CSS so we can drop max_rows param
    css = """
        .container { max-width: 100% !important; }
        .pc-table .wrap.svelte-1ipelgc { height: 240px !important; overflow-y: auto; }
    """

    with gr.Blocks(title="HarmonyScope Viewer", css=css) as demo:
        gr.Markdown("## 🎧 HarmonyScope Interactive Viewer")

        with gr.Row(equal_height=True):
            # ---------- Left side ----------
            with gr.Column(scale=1, min_width=260):
                slider = gr.Slider(
                    0,
                    len(frames) - 1,
                    step=1,
                    value=0,
                    label=f"Frame (hop = {hop:.2f}s)",
                )
                time_box = gr.Textbox(label="Time range", interactive=False)
                chord_box = gr.Textbox(label="Detected chord", interactive=False)
                pc_table = gr.Dataframe(
                    headers=["PC", "Frames"],
                    datatype=["str", "int"],
                    label="Pitch-class summary",
                    wrap=True,
                    elem_classes=["pc-table"],
                )

            # ---------- Right side ----------
            with gr.Column(scale=3):
                wave_plot = gr.Plot(label="Waveform")
                spec_plot = gr.Plot(label="Spectrogram")
                chroma_plot = gr.Plot(label="Chroma")

        # Connect interaction
        slider.change(
            render,
            inputs=slider,
            outputs=[wave_plot, spec_plot, chroma_plot, chord_box, time_box, pc_table],
            show_progress="minimal",
        )
        render(0)  # initial render

    return demo
