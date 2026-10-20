"""B3: classifier explainers (random-weight model on CPU, no trained model or data needed)."""
import numpy as np
import pytest
import torch

from snakeguard.venom import TorchVenomClassifier, VenomNet
from snakeguard.venom.data import eval_transform
from snakeguard.venom.xai import KernelSHAP, LIME, CropView, VenomScorer, build_classifier_explainers

FAST = {"occlusion": {"patch_sizes": [64], "batch": 32}, "rise": {"n_masks": 64, "batch": 32},
        "surrogate": {"grid": 4, "n_samples": 64, "batch": 32}, "integrated_gradients": {"steps": 4, "batch": 4}}


@pytest.fixture(scope="module")
def clf(tmp_path_factory):
    torch.manual_seed(0)
    m = VenomNet(n_species=6, pretrained=False)
    ck = tmp_path_factory.mktemp("ck") / "best.pt"
    torch.save({"model": m.state_dict(), "n_species": 6, "backbone": "efficientnet_b0", "img_size": 224,
                "threshold": 0.5, "venomous_species": [1, 4], "score": "species"}, ck)
    return TorchVenomClassifier(ck, device="cpu")


@pytest.fixture(scope="module")
def img():
    return np.random.default_rng(0).integers(0, 255, (150, 260, 3), dtype=np.uint8)


@pytest.mark.parametrize("hw", [(150, 260), (300, 200), (224, 224), (97, 400)])
def test_cropview_matches_torchvision(hw):
    """A bright square drawn in the original must land where CropView says (inverse map back onto it)."""
    h, w = hw
    im = np.zeros((h, w, 3), np.uint8)
    y0, x0 = h // 3, w // 3
    im[y0:y0 + h // 4, x0:x0 + w // 4] = 255
    x = eval_transform(224, normalize=False)(im)[0].numpy()        # crop as the model sees it
    v = CropView.of(h, w, 224)
    back = v.map_to_orig((x > 0.5).astype(np.float32)) > 0.5
    seen = v.map_to_orig(np.ones((224, 224), np.float32)) > 0.5   # part of the original inside the centre crop
    sq = (im[..., 0] > 0) & seen
    inter, union = (back & sq).sum(), (back | sq).sum()
    assert sq.any() and inter / union > 0.85
    # the crop region itself: centred, full height (or width) of the original
    ys, xs = np.where(seen)
    assert abs((xs.min() + xs.max()) / 2 - (w - 1) / 2) <= 2 and abs((ys.min() + ys.max()) / 2 - (h - 1) / 2) <= 2


def test_scorer_matches_plugin(clf, img):
    x = eval_transform(224, normalize=False)(img)[None]
    s = float(VenomScorer(clf).score(x, 1)[0])
    assert s == pytest.approx(clf.predict(img), abs=1e-5)
    assert float(VenomScorer(clf).score(x, 0)[0]) == pytest.approx(1 - s, abs=1e-5)


@pytest.mark.parametrize("name", ["eigencam", "gradcam", "gradcampp", "gradcampp_ms", "layercam_ms",
                                  "integrated_gradients", "occlusion", "rise", "lime", "kernelshap", "random"])
def test_explainers_shape_range(clf, img, name):
    ex = build_classifier_explainers(clf, FAST, [name])[name]
    assert ex.stage == "classifier" and ex.name == name
    x, view = ex.preprocess(img)
    m = ex.explain_x(x, 1)
    assert m.shape == (224, 224) and m.dtype == np.float32 and 0 <= m.min() and m.max() <= 1
    full = ex.explain(img, 1)
    assert full.shape == img.shape[:2] and 0 <= full.min() and full.max() <= 1
    # centre crop of a wide image: the left/right margins are outside the model's view -> 0
    assert full[:, :5].max() == 0 and full[:, -5:].max() == 0


def test_target_changes_gradient_maps(clf, img):
    ex = build_classifier_explainers(clf, FAST, ["layercam_ms"])["layercam_ms"]
    x, _ = ex.preprocess(img)
    assert not np.allclose(ex.explain_x(x, 1), ex.explain_x(x, 0))


def test_no_param_grads(clf, img):
    ex = build_classifier_explainers(clf, FAST, ["gradcampp"])["gradcampp"]
    ex.explain_x(ex.preprocess(img)[0], 1)
    assert all(p.grad is None for p in clf.model.parameters())


def _additive(z, w):
    return z.astype(np.float64) @ w


def test_kernelshap_recovers_additive_model(clf):
    """For an additive f, Shapley values equal the per-feature weights (and sum to f(1) - f(0))."""
    ks = KernelSHAP(clf, grid=4, n_samples=400)
    w = np.random.default_rng(1).normal(size=16)
    z = ks._coalitions(np.random.default_rng(0))
    phi = ks._fit(z, _additive(z, w) + 0.3)
    np.testing.assert_allclose(phi, w, atol=1e-6)
    assert phi.sum() == pytest.approx(w.sum(), abs=1e-6)


def test_lime_ranks_additive_model(clf):
    lime = LIME(clf, grid=4, n_samples=400)
    w = np.random.default_rng(2).normal(size=16)
    z = lime._coalitions(np.random.default_rng(0))
    coef = lime._fit(z, _additive(z, w))
    assert np.corrcoef(coef, w)[0, 1] > 0.99
