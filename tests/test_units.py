"""Fast tests that need no model."""
import numpy as np
import pytest
import torch

from snakeguard.data import parse_label_file, polygon_mask
from snakeguard.detection.yolo import box_iou_one, letterbox
from snakeguard.eval import metrics as M
from snakeguard.xai.base import normalize


@pytest.mark.parametrize("rect", [True, False])
@pytest.mark.parametrize("hw", [(480, 640), (640, 480), (1000, 333), (640, 640), (17, 900), (500, 375)])
def test_letterbox_roundtrip(hw, rect):
    img = np.random.default_rng(0).integers(0, 255, (*hw, 3), dtype=np.uint8)
    out, lb = letterbox(img, 640, rect=rect)
    assert out.shape == (lb.H, lb.W, 3)
    if rect:
        assert lb.H % 32 == 0 and lb.W % 32 == 0 and max(lb.H, lb.W) == 640
    else:
        assert (lb.H, lb.W) == (640, 640)
    x0, y0, x1, y1 = lb.valid
    assert (x1 - x0, y1 - y0) == (lb.new_w, lb.new_h)
    assert max(lb.new_w, lb.new_h) == 640
    assert (out[:y0] == 114).all() and (out[y1:] == 114).all()
    box = np.array([0.1 * hw[1], 0.2 * hw[0], 0.7 * hw[1], 0.9 * hw[0]])
    np.testing.assert_allclose(lb.box_to_orig(lb.box_to_lb(box)), box, atol=1e-6)
    sal = np.zeros(lb.shape, np.float32); sal[y0:y1, x0:x1] = 1
    back = lb.map_to_orig(sal)
    assert back.shape == hw and np.allclose(back, 1)


@pytest.mark.parametrize("rect", [True, False])
@pytest.mark.parametrize("hw", [(480, 640), (500, 375), (1000, 333), (37, 900)])
def test_letterbox_matches_ultralytics(hw, rect):
    from ultralytics.data.augment import LetterBox
    img = np.random.default_rng(1).integers(0, 255, (*hw, 3), dtype=np.uint8)
    ours, _ = letterbox(img, 640, rect=rect)
    ref = LetterBox((640, 640), auto=rect, stride=32)(image=img)
    assert ours.shape == ref.shape
    assert np.abs(ours.astype(int) - ref.astype(int)).max() <= 1


def test_parse_labels_box_and_polygon(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n\n")
    objs = parse_label_file(f, w=200, h=100)
    assert len(objs) == 2
    np.testing.assert_allclose(objs[0].box, [80, 30, 120, 70])
    assert objs[0].polygon is None
    assert objs[1].cls == 1 and objs[1].polygon.shape == (4, 2)
    np.testing.assert_allclose(objs[1].box, [20, 10, 60, 50])
    m = polygon_mask(objs[1].polygon, (100, 200))
    assert m[30, 40] and not m[5, 5]


def test_iou():
    boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=torch.float32)
    iou = box_iou_one(boxes, torch.tensor([0, 0, 10, 10.]))
    np.testing.assert_allclose(iou.numpy(), [1, 25 / 175, 0], atol=1e-6)


def test_normalize_respects_valid():
    s = np.arange(16, dtype=np.float32).reshape(4, 4)
    v = np.zeros((4, 4), bool); v[1:3, 1:3] = True
    n = normalize(s, v)
    assert n[~v].sum() == 0 and n[v].min() == 0 and n[v].max() == 1


def test_localization_metrics():
    gt = np.zeros((100, 100), bool); gt[40:60, 40:60] = True
    valid = np.ones_like(gt)
    sal = gt.astype(np.float32)
    assert M.pointing_game(sal, gt) == 1
    assert M.energy_pointing_game(sal, gt) == 1
    assert M.saliency_iou(sal, gt, valid) == pytest.approx(1)
    far = np.zeros_like(sal); far[0, 0] = 1
    assert M.pointing_game(far, gt) == 0 and M.energy_pointing_game(far, gt) == 0
    near = np.zeros_like(sal); near[40, 30] = 1          # 10 px left of GT, inside the 15 px tolerance
    assert M.pointing_game(near, gt, tolerance=15) == 1
    assert M.pointing_game(near, gt, tolerance=5) == 0


def test_agreement_metrics():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)); valid = np.ones((64, 64), bool)
    assert M.rank_correlation(a, a, valid) == pytest.approx(1)
    assert M.rank_correlation(a, -a, valid) == pytest.approx(-1)
    assert M.topk_iou(a, a, valid) == 1


def test_rank_map_order_and_ties():
    sal = np.array([[0.1, 0.9], [0.5, 0.5]])
    valid = np.array([[True, True], [True, False]])
    r = M._rank_map(sal, valid, np.random.default_rng(0))
    assert r[0, 1] == 0 and r[1, 0] == 1 and r[0, 0] == 2 and np.isinf(r[1, 1])


class _FakeDet:
    """Score = mean brightness of a fixed 'object' region, so the ideal explanation is known."""
    def __init__(self, region):
        self.region = region

    def score(self, x, target, batch=64, mode="similarity"):
        y0, y1, x0, x1 = self.region
        return x[:, :, y0:y1, x0:x1].mean((1, 2, 3))


def test_deletion_insertion_prefers_correct_map():
    from snakeguard.detection.yolo import Letterbox
    S = 64
    lb = Letterbox(1.0, 0, 0, S, S, S, S, S, S)
    x = torch.zeros(1, 3, S, S); x[:, :, 20:30, 20:30] = 1.0
    det = _FakeDet((20, 30, 20, 30))
    good = np.zeros((S, S), np.float32); good[20:30, 20:30] = 1
    bad = np.zeros((S, S), np.float32); bad[40:50, 40:50] = 1
    kw = dict(steps=20, deletion_fill="black", insertion_start="black")
    rg = M.deletion_insertion(det, x, lb, None, good, **kw)
    rb = M.deletion_insertion(det, x, lb, None, bad, **kw)
    assert rg["deletion"].auc < rb["deletion"].auc
    assert rg["insertion"].auc > rb["insertion"].auc
    assert rg["deletion"].scores[0] == pytest.approx(1) and rg["deletion"].scores[-1] == pytest.approx(0)
    assert rg["insertion"].scores[-1] == pytest.approx(1)
