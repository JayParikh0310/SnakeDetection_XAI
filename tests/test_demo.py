"""B5: demo logic (src/snakeguard/demo.py) and the Gradio app wiring. Uses best.pt (in the repo) on CPU; no dataset."""
import sys

import cv2
import numpy as np
import pytest
import torch

from snakeguard import PROJECT_ROOT
from snakeguard.demo import DISCLAIMER, SnakeGuardDemo, SnakeInfo, venom_label, verdict_text
from snakeguard.venom import TorchVenomClassifier, VenomNet


def _info(label, threat=50.0):
    return SnakeInfo(np.array([0, 0, 10, 10.]), 0.9, {"venomous": 0.9, "harmless": 0.01, "unknown": None}[label],
                     label, 0.5, threat)


def test_venom_label_and_verdicts():
    assert venom_label(None, 0.14) == "unknown"
    assert venom_label(0.2, 0.14) == "venomous" and venom_label(0.1, 0.14) == "harmless"
    assert verdict_text([], 0) == "No snake detected."
    assert "no alert" in verdict_text([], 2)
    assert "possibly venomous" in verdict_text([_info("venomous")], 0)
    assert "treated as venomous" in verdict_text([_info("unknown")], 0)          # CLAUDE.md rule 9
    assert "harmless" in verdict_text([_info("harmless")], 0) and "distance" in verdict_text([_info("harmless")], 0)
    assert "waiting for confirmation" in verdict_text([_info("venomous")], 0, confirmed=False)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    torch.manual_seed(0)
    ck = tmp_path_factory.mktemp("ck") / "best.pt"
    m = VenomNet(n_species=4, pretrained=False)
    torch.save({"model": m.state_dict(), "n_species": 4, "backbone": "efficientnet_b0", "img_size": 224,
                "threshold": 0.3, "venomous_species": [1], "score": "species"}, ck)
    return SnakeGuardDemo(device="cpu", classifier=TorchVenomClassifier(ck, device="cpu"))


def test_photo_without_snake_has_disclaimer(demo):
    img = np.full((240, 320, 3), 128, np.uint8)
    r = demo.analyze_photo(img)
    assert r.annotated.shape == img.shape
    assert r.verdict == "No snake detected." and r.threat == 0 and r.beep_interval_ms is None
    assert DISCLAIMER in r.details_md


def test_photo_on_real_image_if_available(demo):
    """Smoke test on a dataset image when B's SnakeCLEF data is present (skipped otherwise)."""
    import pandas as pd
    pred = PROJECT_ROOT / "results" / "venom_clf" / "b0_reg" / "predictions_test.csv"
    root = PROJECT_ROOT / "data" / "snakeclef" / "images"
    if not pred.exists() or not root.exists():
        pytest.skip("SnakeCLEF data not available")
    p = pd.read_csv(pred).image_path.iloc[0]
    img = cv2.cvtColor(cv2.imread(str(root / p)), cv2.COLOR_BGR2RGB)
    r = demo.analyze_photo(img)
    assert r.annotated.shape == img.shape and DISCLAIMER in r.details_md
    if r.snakes:
        assert r.detector_map is not None and r.detector_map.shape == img.shape
        assert r.venom_map is not None and 0 <= r.threat <= 100


def test_video_roundtrip(demo, tmp_path):
    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"avc1"), 10, (160, 120))
    for i in range(20):
        w.write(np.full((120, 160, 3), 40 + 5 * i, np.uint8))
    w.release()
    out, rows = demo.analyze_video(str(src), str(tmp_path / "out.mp4"), target_fps=5)
    assert len(rows) == 10 and all(r["snakes"] == 0 and r["threat"] == 0 for r in rows)
    cap = cv2.VideoCapture(out)
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_app_builds(demo):
    pytest.importorskip("gradio")
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from app import build
    import gradio as gr
    assert isinstance(build(demo), gr.Blocks)


# ------------------------------------------------------------------ crop context + rescaled venom score (2026-10-20)
def test_venom_crop_margin_zero_matches_old_crop():
    from snakeguard.pipeline import venom_crop
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
    for _ in range(200):
        box = rng.uniform(-30, 230, 4); box[2:] = np.maximum(box[2:], box[:2] + 1)
        x0, y0, x1, y1 = np.round(box).astype(int)
        old = frame[max(y0, 0):y1, max(x0, 0):x1]               # the pipeline's original crop
        assert np.array_equal(venom_crop(frame, box, 0.0), old)


def test_venom_crop_margin_grows_and_clips():
    from snakeguard.pipeline import venom_crop
    frame = np.zeros((100, 100, 3), np.uint8)
    assert venom_crop(frame, [40, 40, 60, 60], 0.25).shape[:2] == (30, 30)   # 20 px box + 5 px per side
    assert venom_crop(frame, [0, 0, 50, 50], 0.5).shape[:2] == (75, 75)      # clipped at the frame edge


def test_pipeline_passes_margin_crop_to_classifier():
    from snakeguard.detection.yolo import Detection, letterbox
    from snakeguard.pipeline import PipelineConfig, SnakeGuard, VenomClassifier

    class FakeDet:
        def preprocess(self, img):
            return torch.zeros(1, 3, 64, 64), letterbox(img, 64)[1]

        def detect_tensor(self, x, lb, conf, iou):
            b = np.array([40, 40, 60, 60.])
            return [Detection(b, b, 0, 0.9, 0)]

    class Rec(VenomClassifier):
        shapes = []

        def predict(self, crop):
            self.shapes.append(crop.shape[:2]); return 0.9
    for margin, hw in ((0.0, (20, 20)), (0.25, (30, 30))):
        clf = Rec(); clf.shapes = []
        SnakeGuard(FakeDet(), PipelineConfig(k_confirm=1, venom_crop_margin=margin), classifier=clf).process(
            np.zeros((100, 100, 3), np.uint8))
        assert clf.shapes == [hw]


def test_rescale_score_maps_threshold_to_half():
    from snakeguard.venom.classifier import rescale_score
    thr = 0.1413
    assert rescale_score(thr, thr) == pytest.approx(0.5)
    assert rescale_score(0.0, thr) == 0.0 and rescale_score(1.0, thr) == pytest.approx(1.0)
    ps = np.linspace(0, 1, 101)
    out = [rescale_score(p, thr) for p in ps]
    assert all(b >= a for a, b in zip(out, out[1:]))                   # monotone: rankings unchanged
    assert all((p >= thr) == (o >= 0.5) for p, o in zip(ps, out))      # same decisions at 0.5


def test_classifier_rescale_option(tmp_path):
    from snakeguard.pipeline import PipelineConfig, threat_score
    from snakeguard.venom.classifier import rescale_score
    torch.manual_seed(0)
    m = VenomNet(n_species=4, pretrained=False)
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "n_species": 4, "backbone": "efficientnet_b0", "img_size": 224,
                "threshold": 0.14, "venomous_species": [1], "score": "species"}, ck)
    raw, res = TorchVenomClassifier(ck, device="cpu"), TorchVenomClassifier(ck, device="cpu", rescale=True)
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert raw.threshold == pytest.approx(0.14) and res.threshold == 0.5 and res.raw_threshold == pytest.approx(0.14)
    assert res.predict(img) == pytest.approx(rescale_score(raw.predict(img), 0.14), abs=1e-6)
    # the point of rescaling: a snake just over the venom threshold now adds a real share of threat
    c = PipelineConfig()
    harmless, venomous_raw = threat_score(0.9, 0.5, 0.0, c), threat_score(0.9, 0.5, 0.15, c)
    venomous_rescaled = threat_score(0.9, 0.5, rescale_score(0.15, 0.14), c)
    assert venomous_raw - harmless < 6 and venomous_rescaled - harmless > 18
