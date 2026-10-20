"""B4: classifier XAI benchmark helpers (no trained model or data needed)."""
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from snakeguard import PROJECT_ROOT
from snakeguard.detection.yolo import Letterbox
from snakeguard.eval import metrics as M
from snakeguard.venom import VenomNet
from snakeguard.venom.xai import CropView

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_venom_xai_bench import RandomizedClassifier, every_kth, range_auc, sample_images  # noqa: E402


@pytest.mark.parametrize("level", ["head", "top", "all"])
def test_randomized_classifier_restores(level):
    torch.manual_seed(0)
    m = VenomNet(n_species=5, pretrained=False).eval()
    x = torch.rand(1, 3, 64, 64)
    before = m(x)[1].detach().clone()
    head_w = m.venom_head.weight.detach().clone()
    stem_w = m.features[0][0].weight.detach().clone()
    with RandomizedClassifier(m, level):
        assert not torch.allclose(m(x)[1], before)
        assert not torch.equal(m.venom_head.weight, head_w)
        assert torch.equal(m.features[0][0].weight, stem_w) == (level != "all")  # stem only randomized at "all"
    torch.testing.assert_close(m(x)[1], before)


def test_sample_images_kinds_and_observations():
    rng = np.random.default_rng(0)
    n = 400
    pred = pd.DataFrame({"venomous": rng.integers(0, 2, n), "s": rng.random(n),
                         "observation_id": [str(i // 2) for i in range(n)], "image_path": [f"{i}.jpg" for i in range(n)]})
    s = sample_images(pred, 0.5, {"TP": 10, "FP": 10, "FN": 3, "TN": 3}, seed=1)
    assert s.kind.value_counts().to_dict() == {"TP": 10, "FP": 10, "FN": 3, "TN": 3}
    assert s.observation_id.is_unique
    tp = s[s.kind == "TP"]
    assert ((tp.venomous == 1) & (tp.s >= 0.5)).all()


def test_every_kth_and_range_auc():
    ids = [str(i) for i in range(100)]
    assert len(every_kth(ids, 10)) == 10 and every_kth(ids, 0) == set()
    fr = np.linspace(0, 1, 11)
    assert range_auc(fr, np.full(11, 0.9), 0.9, 0.1) == pytest.approx(1.0)   # never drops
    assert range_auc(fr, np.full(11, 0.1), 0.9, 0.1) == pytest.approx(0.0)   # at blank immediately
    assert np.isnan(range_auc(fr, np.full(11, 0.5), 0.5, 0.5))              # no gap -> undefined


def test_box_to_crop_and_deletion_with_identity_letterbox():
    v = CropView.of(150, 300, 224)                     # wide image: crop drops left/right margins
    m = v.box_mask([0, 0, 300, 150])                   # whole image -> whole crop
    assert m.all()
    m2 = v.box_mask([0, 0, 40, 150])                   # far-left strip is outside the crop
    assert not m2.any()

    class Score:  # mean brightness of the left half
        def score(self, x, target, batch=64, mode=None):
            return x[:, :, :, :32].mean((1, 2, 3))
    S = 64
    lb = Letterbox(1.0, 0, 0, S, S, S, S, S, S)
    x = torch.zeros(1, 3, S, S); x[..., :32] = 1
    good = np.zeros((S, S), np.float32); good[:, :32] = 1
    r = M.deletion_insertion(Score(), x, lb, 1, good, steps=10, deletion_fill="black", insertion_start="black")
    rb = M.deletion_insertion(Score(), x, lb, 1, 1 - good, steps=10, deletion_fill="black", insertion_start="black")
    assert r["deletion"].auc < rb["deletion"].auc


def test_wilson_interval():
    from venom_species_report import wilson
    lo, hi = wilson(12, 16)
    assert 0 < lo < 12 / 16 < hi < 1
    assert wilson(16, 16)[1] == pytest.approx(1.0) and wilson(0, 16)[0] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(wilson(0, 0)[0])
    assert wilson(120, 160)[1] - wilson(120, 160)[0] < hi - lo   # more data -> narrower
