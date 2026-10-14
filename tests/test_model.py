"""Integration tests against the real best.pt + dataset (marked gpu; they also run on CPU, slowly)."""
import numpy as np
import pytest
import torch

from snakeguard import load_config
from snakeguard.data import list_split, load_sample
from snakeguard.detection import YoloDetector
from snakeguard.eval import metrics as M
from snakeguard.xai import build_explainers

pytestmark = pytest.mark.gpu
CFG = load_config()


@pytest.fixture(scope="module")
def det():
    return YoloDetector(CFG["weights"])


@pytest.fixture(scope="module")
def case(det):
    """First test image that has a confident detection."""
    for p in list_split(CFG["data_root"], "test")[:50]:
        s = load_sample(p)
        img = s.load_rgb()
        d = det.detect(img, conf=0.5)
        if d:
            return img, d[0]
    pytest.skip("no confident detection found")


def test_model_metadata(det):
    assert det.names == {0: "snake", 1: "snake_like_object"}
    assert len(det.scale_layers) == 3


def test_matches_ultralytics_predict(det):
    """Our letterbox + NMS must reproduce ultralytics' own predictions (same boxes/classes)."""
    paths = list_split(CFG["data_root"], "test")[:20]
    for p in paths:
        img = load_sample(p).load_rgb()
        ours = det.detect(img, conf=0.25)
        ref = det.yolo.predict(img[..., ::-1].copy(), conf=0.25, imgsz=640, verbose=False)[0]
        ref_boxes = ref.boxes.xyxy.cpu().numpy()
        ref_cls = ref.boxes.cls.cpu().numpy().astype(int)
        strong_ours = [d for d in ours if d.score > 0.35]
        for d in strong_ours:  # every confident detection of ours must exist in ultralytics' output
            assert len(ref_boxes), p.name
            ious = [_iou(d.box, b) for b in ref_boxes]
            j = int(np.argmax(ious))
            assert ious[j] > 0.9 and ref_cls[j] == d.cls, (p.name, d, ref_boxes)


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def test_target_score_equals_detection_score(det, case):
    img, d = case
    x, lb = det.preprocess(img)
    s = float(det.score(x, d)[0])
    assert s == pytest.approx(d.score, abs=1e-4)  # IoU with itself = 1 and nothing scores higher


@pytest.mark.parametrize("name", ["eigencam", "eigencam_p5", "eigencam_centered", "gradcam", "gradcampp",
                                  "gradcampp_p5", "gradcampp_contrast", "layercam", "layercam_fsum", "occlusion", "random", "boxprior"])
def test_explainers_shape_range(det, case, name):
    img, d = case
    ex = build_explainers(det, {"occlusion": {"patch_sizes": [96]}}, [name])[name]
    x, lb = det.preprocess(img)
    m = ex.explain_lb(x, lb, d)
    assert m.shape == lb.shape and m.dtype == np.float32
    assert 0 <= m.min() and m.max() <= 1
    # P5-only Grad-CAM++ is legitimately empty on some detections (all channel-weighted activations
    # negative). pytorch-grad-cam gives the same empty map. This is a finding, not a bug.
    if name != "gradcampp_p5":
        assert m.max() > 0, "map must not be empty"
    assert m[~lb.valid_mask()].sum() == 0
    full = ex.explain(img, d)
    assert full.shape == img.shape[:2]


def test_eigencam_sign_regression(det):
    """snake_021130: the raw P5 projection is all positive while the mean activation is partly
    negative. The old sign heuristic flipped it and produced an empty map."""
    p = [q for q in list_split(CFG["data_root"], "test") if q.stem == "snake_021130"][0]
    img = load_sample(p).load_rgb()
    d = det.detect(img, conf=0.25)[0]
    x, lb = det.preprocess(img)
    for n in ["eigencam", "eigencam_p5", "eigencam_centered"]:
        assert build_explainers(det, names=[n])[n].explain_lb(x, lb, d).max() > 0, n


def test_drise_small(det, case):
    img, d = case
    ex = build_explainers(det, {"drise": {"n_masks": 128}}, ["drise"])["drise"]
    x, lb = det.preprocess(img)
    m = ex.explain_lb(x, lb, d)
    assert m.max() > 0


def test_gradients_flow_and_no_param_grads(det, case):
    img, d = case
    ex = build_explainers(det, names=["gradcampp"])["gradcampp"]
    x, lb = det.preprocess(img)
    ex.explain_lb(x, lb, d)
    assert all(p.grad is None for p in det.model.parameters())


def test_deletion_insertion_real(det, case):
    img, d = case
    x, lb = det.preprocess(img)
    occ = build_explainers(det, {"occlusion": {"patch_sizes": [96]}}, ["occlusion"])["occlusion"]
    rnd = build_explainers(det, names=["random"])["random"]
    r_occ = M.deletion_insertion(det, x, lb, d, occ.explain_lb(x, lb, d), steps=20)
    r_rnd = M.deletion_insertion(det, x, lb, d, rnd.explain_lb(x, lb, d), steps=20)
    assert r_occ["deletion"].scores[0] == pytest.approx(d.score, abs=1e-4)
    assert r_occ["deletion"].auc < r_rnd["deletion"].auc  # occlusion should beat random on deletion


def test_randomization_restores_weights(det, case):
    img, d = case
    x, lb = det.preprocess(img)
    before = float(det.score(x, d)[0])
    with M.RandomizedDetector(det, down_to=det.scale_layers[0]):
        changed = float(det.score(x, d)[0])
    after = float(det.score(x, d)[0])
    assert after == pytest.approx(before, abs=1e-6)
    assert abs(changed - before) > 1e-3


def test_pipeline_end_to_end_with_layercam(det, case):
    """The usage recommended to Member B in PROGRESS.md: SnakeGuard + LayerCAM on a real image."""
    from snakeguard.pipeline import PipelineConfig, SnakeGuard
    img, _ = case
    g = SnakeGuard(det, PipelineConfig(k_confirm=1), explainer=build_explainers(det, names=["layercam"])["layercam"])
    r = g.process(img, explain=True)
    assert r.detections and r.confirmed == bool(r.snakes)
    assert 0 <= r.threat <= 100
    sal = r.explanation["saliency"]
    assert sal.shape == img.shape[:2] and 0 <= sal.min() and sal.max() <= 1
    assert 0 <= r.explanation["mass_in_box"] <= 1
