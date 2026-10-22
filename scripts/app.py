"""B5: SnakeGuard demo app (Gradio). Logic lives in src/snakeguard/demo.py.
Tabs: Photo, Video, Live (webcam stream with 3-of-5 confirmation, threat HUD and browser beeps; added 2026-10-22).

Usage:  .venv/bin/python scripts/app.py [--venom results/venom_clf/b0_reg/best.pt] [--port 7860] [--share]
Then open http://127.0.0.1:7860
"""
from __future__ import annotations

import argparse
import json
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


LIVE_EVERY_S = 0.1   # browser sends a webcam frame every 100 ms; frames arriving while the GPU is busy are dropped

LIVE_HOW = """**Live camera.** Click the camera, allow access, then press the record button to start streaming. Every frame
goes through the same pipeline as the ankle camera: detection → venom check → **3-of-5 frame confirmation** → threat → beep.
Bring a snake picture closer and the box grows, so proximity, threat and the beep rate go up until the beeps merge into
one **continuous tone** (threat ≥ 65); a venomous call beeps at a higher pitch. Click anywhere on the page once so the browser allows sound."""

# Browser beeper: Web Audio square-wave beeps at demo.live_beep_ms (0 = continuous tone). Fed by the hidden textbox below; goes silent
# when the stream stops sending (no update for 1.5 s), when threat is 0, or when "Beep" is unticked.
BEEP_JS = """<script>
(() => {
  const S = {interval: null, venom: false, on: true, last: 0, seen: 0, ctx: null, count: 0, continuous: false};
  window.sgBeepState = S;
  let tone = null;
  function audio() {
    if (!S.ctx) { const C = window.AudioContext || window.webkitAudioContext; if (C) S.ctx = new C(); }
    if (S.ctx && S.ctx.state === "suspended") S.ctx.resume();
    return S.ctx;
  }
  ["pointerdown", "keydown"].forEach((e) => document.addEventListener(e, audio, {capture: true}));
  function beep(venom, interval) {
    const c = audio(); if (!c || c.state !== "running") return;
    const o = c.createOscillator(), g = c.createGain(), t = c.currentTime;
    const dur = Math.min(0.1, 0.55 * interval / 1000);   // short gaps at fast rates -> buzz
    o.type = "square"; o.frequency.value = venom ? 1400 : 880;
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.2, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(c.destination); o.start(t); o.stop(t + dur + 0.01);
    S.count += 1;
  }
  function toneOn(venom) {                               // the continuous "beeeeee"
    const c = audio(); if (!c || c.state !== "running") return;
    const f = venom ? 1400 : 880;
    if (tone) { tone.o.frequency.value = f; return; }
    const o = c.createOscillator(), g = c.createGain(), t = c.currentTime;
    o.type = "square"; o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.15, t + 0.02);
    o.connect(g).connect(c.destination); o.start(t);
    tone = {o, g}; S.continuous = true; S.count += 1;
  }
  function toneOff() {
    if (!tone) return;
    const t = S.ctx.currentTime;
    tone.g.gain.cancelScheduledValues(t); tone.g.gain.setValueAtTime(0.15, t);
    tone.g.gain.exponentialRampToValueAtTime(0.0001, t + 0.03); tone.o.stop(t + 0.05);
    tone = null; S.continuous = false;
  }
  window.sgBeepSet = (s, on) => {
    try { const d = JSON.parse(s || "{}"); S.interval = d.beep_ms === 0 ? 0 : (d.beep_ms || null); S.venom = !!d.venom; }
    catch (e) { S.interval = null; }
    S.on = on !== false; S.seen = performance.now();
  };
  setInterval(() => {
    const now = performance.now();
    if (!S.on || S.interval === null || now - S.seen > 1500) { toneOff(); return; }   // muted / safe / stream stopped
    if (S.interval === 0) { toneOn(S.venom); return; }
    toneOff();
    if (now - S.last >= S.interval) { S.last = now; beep(S.venom, S.interval); }
  }, 15);
})();
</script>"""


def live_panel(info: dict | None) -> str:
    """Big live numbers under the video (inline styles: readable in light and dark themes)."""
    fg = "color:#f2f5f3 !important;margin:0;line-height:1.3"   # !important: Gradio's prose styles recolor text
    box = f"background:#1f2622;border-radius:10px;padding:10px 12px;{fg}"
    tile = f"flex:1 1 150px;{box}"
    lab = f"font-size:12px;opacity:.75;{fg}"
    val = f"font-size:18px;font-weight:600;{fg}"
    if info is None:
        return (f'<div style="{box}"><div style="{lab}">Status</div>'
                f'<div style="{val}">Waiting for the camera… (click the camera, then Record)</div></div>')
    t = info["threat"]
    col = "#2e9e44" if t < 35 else "#e0a020" if t < 65 else "#d63a2f"
    if info["snakes"] and info["confirmed"]:
        status, scol = ("SNAKE: possibly venomous" if info["venom_label"] in ("venomous", "unknown")
                        else "SNAKE: likely harmless"), col
    elif info["snakes"]:
        status, scol = f"Snake? confirming ({info['seen']}/{info['window']}, need {info['k']})", "#e0a020"
    elif info["lookalikes"]:
        status, scol = f"Look-alike only ({info['lookalikes']}): no alert", "#3b7dd8"
    else:
        status, scol = "Clear", "#2e9e44"
    prox = info["proximity"]
    near = "–" if prox is None else ("near" if prox >= 0.66 else "medium" if prox >= 0.33 else "far")
    venom = "–" if not info["snakes"] else (f"{info['venom_label']} ({info['venom_score']:.2f})"
                                            if info["venom_score"] is not None else "unknown → treated as venomous")
    beep = ("silent" if info["beep_ms"] is None else "CONTINUOUS TONE" if info["beep_ms"] == 0
            else f"every {info['beep_ms']} ms")
    cells = [("Detector confidence", "–" if info["det_score"] is None else f"{info['det_score']:.2f}"),
             ("Venom status", venom), ("Proximity", "–" if prox is None else f"{prox:.2f} ({near})"),
             ("Beep", beep), ("Confirmation", f"{info['seen']}/{info['window']} frames"),
             ("Speed", f"{info['latency_ms']} ms/frame · {info['fps']} fps · {info.get('size', '')}")]
    if info["mass_in_box"] is not None:
        cells.append(("Heatmap inside box", f"{100 * info['mass_in_box']:.0f} %"))
    tiles = "".join(f'<div style="{tile}"><div style="{lab}">{k}</div><div style="{val}">{v}</div></div>'
                    for k, v in cells)
    return (f'<div style="display:flex;flex-direction:column;gap:8px">'
            f'<div style="{box};border-left:8px solid {scol}"><div style="{lab}">Status</div>'
            f'<div style="{val};font-size:22px;font-weight:700">{status}</div></div>'
            f'<div style="{box}"><div style="{lab}">Threat</div><div style="display:flex;align-items:center;gap:12px">'
            f'<div style="{val};font-size:30px;font-weight:800;min-width:90px">{t:.0f}<span style="{fg};font-size:15px">/100</span></div>'
            f'<div style="flex:1;height:16px;background:#3a423d;border-radius:8px;overflow:hidden">'
            f'<div style="width:{t:.0f}%;height:100%;background:{col}"></div></div></div></div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:8px">{tiles}</div></div>')


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

    def run_live(frame, heat, sess):
        if sess is None:
            sess = demo.live_session()
        if frame is None:
            return gr.skip(), gr.skip(), gr.skip(), sess
        out, info = sess.step(frame, heatmap=bool(heat))
        bus = json.dumps({"n": info["frame"], "beep_ms": info["beep_ms"], "venom": info["venom_alert"]})
        return out, live_panel(info), bus, sess

    def reset_live(sess):
        if sess is not None:
            sess.reset()
        return sess, live_panel(None), json.dumps({"n": -1, "beep_ms": None})

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
        with gr.Tab("Live"):
            gr.Markdown(LIVE_HOW)
            with gr.Row():
                with gr.Column(scale=1):
                    cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
                    heat = gr.Checkbox(label="Show detector heatmap (LayerCAM) on the live video", value=False)
                    sound = gr.Checkbox(label="Beep", value=True)
                    lreset = gr.Button("Reset confirmation")
                with gr.Column(scale=2):
                    live_out = gr.Image(label="Live analysis", show_label=False, interactive=False)
                    live_stats = gr.HTML(live_panel(None))
            beep_bus = gr.Textbox(visible="hidden", elem_id="sg-beep-bus")   # carries threat/beep to the browser
            session = gr.State(None)
            cam.stream(run_live, [cam, heat, session], [live_out, live_stats, beep_bus, session],
                       stream_every=LIVE_EVERY_S, concurrency_limit=1, show_progress="hidden")
            beep_bus.change(None, [beep_bus, sound], None, js="(s, on) => { window.sgBeepSet && window.sgBeepSet(s, on); }")
            sound.change(None, [beep_bus, sound], None, js="(s, on) => { window.sgBeepSet && window.sgBeepSet(s, on); }")
            lreset.click(reset_live, [session], [session, live_stats, beep_bus])
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
    build(demo).launch(server_name="127.0.0.1", server_port=a.port, share=a.share, head=BEEP_JS)


if __name__ == "__main__":
    main()
