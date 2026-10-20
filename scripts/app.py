"""B5: SnakeGuard demo app (Gradio). Logic lives in src/snakeguard/demo.py.

Usage:  .venv/bin/python scripts/app.py [--venom results/venom_clf/b0_reg/best.pt] [--port 7860] [--share]
Then open http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from snakeguard import PROJECT_ROOT
from snakeguard.demo import DISCLAIMER, SnakeGuardDemo

DEFAULT_VENOM = PROJECT_ROOT / "results" / "venom_clf" / "b0_reg" / "best.pt"

INTRO = """# SnakeGuard
Ankle-camera snake early warning: **detect** snakes (and tell them apart from ropes, roots, hoses and sticks),
estimate whether the snake is **medically important venomous**, score the **threat**, and **explain** each decision."""

HOW = """**How to read the result**
- Red box = snake, blue box = snake-like object (no alert).
- *Threat 0–100* combines detector confidence, distance (box size) and venom status; it sets the beep rate.
- *Venom status* uses a threshold tuned to catch 95 % of venomous snakes, so harmless snakes are often flagged too.
- The heatmaps show where each model looked (red = most important). *Audit mode* uses slower, perturbation-based
  methods (occlusion / RISE) as a cross-check."""


def build(demo: SnakeGuardDemo) -> gr.Blocks:
    def run_photo(img, audit):
        if img is None:
            return None, None, None, "Upload a photo first.\n\n" + DISCLAIMER
        r = demo.analyze_photo(img, explain=True, audit=bool(audit))
        return r.annotated, r.detector_map, r.venom_map, r.details_md

    def run_video(path):
        if not path:
            return None, None, "Upload a video first.\n\n" + DISCLAIMER
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out, rows = demo.analyze_video(path, out)
        df = pd.DataFrame(rows)
        n_alert = int((df.confirmed & (df.snakes > 0)).sum())
        md = (f"Processed {len(df)} frames (~5 per second, first 30 s). A snake alert needs a snake in 3 of the last 5 "
              f"processed frames. Confirmed alert in **{n_alert}** frames, max threat **{df.threat.max():.0f}/100**."
              f"\n\n{DISCLAIMER}")
        return out, df, md

    with gr.Blocks(title="SnakeGuard") as app:
        gr.Markdown(INTRO)
        with gr.Tab("Photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="numpy", label="Photo (e.g. ankle-camera frame)", sources=["upload", "webcam", "clipboard"])
                    audit = gr.Checkbox(label="Audit mode (slower: occlusion + RISE explanations)", value=False)
                    btn = gr.Button("Analyze", variant="primary")
                    gr.Markdown(HOW)
                with gr.Column(scale=2):
                    out_img = gr.Image(label="Detections", interactive=False)
                    details = gr.Markdown(DISCLAIMER)
                    with gr.Row():
                        det_map = gr.Image(label="Why the detector fired", interactive=False)
                        venom_map = gr.Image(label="Why this venom call (snake crop)", interactive=False)
            btn.click(run_photo, [inp, audit], [out_img, det_map, venom_map, details])
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vin = gr.Video(label="Video (first 30 s are processed)")
                    vbtn = gr.Button("Analyze video", variant="primary")
                with gr.Column(scale=2):
                    vout = gr.Video(label="Annotated video", interactive=False)
                    vmd = gr.Markdown(DISCLAIMER)
                    vtable = gr.Dataframe(label="Timeline", interactive=False)
            vbtn.click(run_video, [vin], [vout, vtable, vmd])
        gr.Markdown(DISCLAIMER)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venom", default=str(DEFAULT_VENOM), help="venom classifier checkpoint (missing -> unknown)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public Gradio link (off by default)")
    a = ap.parse_args()
    demo = SnakeGuardDemo(venom_checkpoint=a.venom, device=a.device)
    print(f"venom classifier: {a.venom if demo.clf is not None else 'none (unknown -> treated as venomous)'}", flush=True)
    build(demo).launch(server_name="127.0.0.1", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
