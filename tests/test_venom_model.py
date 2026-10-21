"""B2: venom model, data loading, classifier plug-in and threshold logic (no trained model needed)."""
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from snakeguard import PROJECT_ROOT
from snakeguard.venom import TorchVenomClassifier, VenomDataset, VenomNet, eval_transform, train_transform

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_venom import binary_stats, pick_threshold  # noqa: E402


def test_venomnet_shapes():
    m = VenomNet(n_species=7, pretrained=False).eval()
    v, s = m(torch.rand(2, 3, 224, 224))
    assert v.shape == (2,) and s.shape == (2, 7)
    assert m.features(torch.rand(1, 3, 224, 224)).shape[1] == 1280


@pytest.mark.parametrize("hw", [(240, 180), (90, 240), (240, 240)])
def test_transforms_shape(hw):
    img = np.random.default_rng(0).integers(0, 255, (*hw, 3), dtype=np.uint8)
    for tf in (train_transform(224), eval_transform(224)):
        x = tf(img)
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_dataset_reads_images(tmp_path):
    import cv2
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.full((50, 80, 3), 40 * i, np.uint8))
    man = pd.DataFrame({"image_path": [f"{i}.jpg" for i in range(3)], "venomous": [0, 1, 0], "species_idx": [0, 2, 1]})
    x, yv, ys = VenomDataset(man, tmp_path, eval_transform(64))[1]
    assert x.shape == (3, 64, 64) and float(yv) == 1 and int(ys) == 2


def test_classifier_plugin(tmp_path):
    m = VenomNet(n_species=5, pretrained=False)
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "n_species": 5, "backbone": "efficientnet_b0", "img_size": 224,
                "threshold": 0.3, "venomous_species": [1, 3]}, ck)
    clf = TorchVenomClassifier(ck, device="cpu")
    p = clf.predict(np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8))
    assert 0 <= p <= 1 and clf.threshold == 0.3
    assert clf.predict(np.zeros((8, 200, 3), np.uint8)) is None  # too small -> unknown (pipeline: venomous)


def test_pick_threshold_meets_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000); p = np.clip(y * 0.3 + rng.random(1000) * 0.7, 0, 1)
    for target in (0.8, 0.95, 1.0):
        thr = pick_threshold(y, p, target)
        st = binary_stats(y, p, thr)
        assert st["recall_venomous"] >= target
        # a slightly higher threshold would drop below the target
        assert binary_stats(y, p, thr + 1e-9)["recall_venomous"] < target or target == 0


def test_pipeline_with_venom_classifier(tmp_path):
    """The classifier's output reaches the pipeline's threat score / venom alert."""
    from snakeguard.detection.yolo import Detection, letterbox
    from snakeguard.pipeline import PipelineConfig, SnakeGuard

    class FakeDet:
        def preprocess(self, img):
            return torch.zeros(1, 3, 64, 64), letterbox(img, 64)[1]

        def detect_tensor(self, x, lb, conf, iou):
            b = np.array([20, 20, 180, 180.])
            return [Detection(b, b, 0, 0.9, 0)]

    m = VenomNet(n_species=3, pretrained=False)
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "n_species": 3, "backbone": "efficientnet_b0", "threshold": 0.0}, ck)
    clf = TorchVenomClassifier(ck, device="cpu")
    g = SnakeGuard(FakeDet(), PipelineConfig(k_confirm=1, venom_threshold=clf.threshold), classifier=clf)
    r = g.process(np.zeros((200, 200, 3), np.uint8))
    assert r.snakes[0].p_venom is not None and r.venomous_alert


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_predict_on_device(device):
    """Regression: the venom mask lives on the CPU while the model may be on MPS/CUDA."""
    from train_venom import predict
    m = VenomNet(n_species=4, pretrained=False).to(device)
    dl = [(torch.rand(3, 3, 64, 64), torch.zeros(3), torch.zeros(3, dtype=torch.long))]
    pv, ps, sp = predict(m, dl, torch.device(device), torch.tensor([0., 1., 0., 1.]))
    assert pv.shape == ps.shape == sp.shape == (3,)
    assert ((0 <= pv) & (pv <= 1)).all() and ((0 <= ps) & (ps <= 1 + 1e-6)).all()


def test_compare_scores_helpers():
    from compare_venom_scores import scores, spec_at_recall
    d = pd.DataFrame({"p_venom": [0.9, 0.8, 0.1, 0.2], "p_venom_species": [0.7, 0.6, 0.3, 0.0]})
    s = scores(d)
    np.testing.assert_allclose(s["mean"], [0.8, 0.7, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    assert spec_at_recall(y, s["venom"], 0.95) == 1.0       # perfect ranking
    assert spec_at_recall(y, np.array([0.9, 0.1, 0.5, 0.2]), 0.95) == 0.0  # must go down to 0.1 to catch both


def test_classifier_species_score(tmp_path):
    """score='species': P(venomous) = sum of species probabilities over the venomous species."""
    torch.manual_seed(0)
    m = VenomNet(n_species=4, pretrained=False).eval()
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "n_species": 4, "backbone": "efficientnet_b0", "img_size": 224,
                "threshold": 0.4, "venomous_species": [1, 3], "score": "species"}, ck)
    clf = TorchVenomClassifier(ck, device="cpu")
    img = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    with torch.no_grad():
        _, s = m(clf.tf(img)[None])
    expected = float(s.softmax(-1)[0, [1, 3]].sum())
    assert clf.score == "species" and clf.predict(img) == pytest.approx(expected, abs=1e-5)
