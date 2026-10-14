"""Pipeline logic tests with a fake detector (no model needed)."""
import numpy as np
import pytest
import torch

from snakeguard.detection.yolo import Detection, letterbox
from snakeguard.pipeline import (PipelineConfig, SnakeGuard, VenomClassifier, beep_interval, proximity,
                                 resolve_conflicts, threat_score)


def D(box, cls, score):
    b = np.array(box, float)
    return Detection(b, b, cls, score, 0)


class FakeDet:
    def __init__(self, frames):
        self.frames = iter(frames)

    def preprocess(self, img):
        lb_img, lb = letterbox(img, 64)
        return torch.zeros(1, 3, 64, 64), lb

    def detect_tensor(self, x, lb, conf, iou):
        return next(self.frames)


def test_conflict_resolution_safety_bias():
    snake, look = D([0, 0, 10, 10], 0, 0.5), D([0, 0, 10, 10], 1, 0.6)
    assert [d.cls for d in resolve_conflicts([snake, look], 0.5, 0.15)] == [0]  # look-alike not clearly better
    look_strong = D([0, 0, 10, 10], 1, 0.9)
    assert [d.cls for d in resolve_conflicts([snake, look_strong], 0.5, 0.15)] == [1]
    far = D([50, 50, 60, 60], 1, 0.9)
    assert len(resolve_conflicts([snake, far], 0.5, 0.15)) == 2


def test_proximity_threat_beep_monotonic():
    c = PipelineConfig()
    assert proximity(0.001, c) == 0 and proximity(0.5, c) == 1
    assert proximity(0.05, c) < proximity(0.1, c)
    assert threat_score(0.9, 1.0, 1.0, c) == pytest.approx(90)
    assert threat_score(0.9, 0.5, None, c) == threat_score(0.9, 0.5, 1.0, c)  # unknown venom = conservative
    assert beep_interval(0, c) is None
    assert beep_interval(100, c) == c.beep_fast_ms and beep_interval(10, c) > beep_interval(90, c)


def test_temporal_confirmation_3_of_5():
    img = np.zeros((100, 100, 3), np.uint8)
    snake = [D([10, 10, 60, 60], 0, 0.9)]
    seq = [snake, [], snake, [], snake, snake, [], [], [], []]
    g = SnakeGuard(FakeDet(seq))
    confirmed = [g.process(img).confirmed for _ in seq]
    # window after each frame: ..., f4 [1,0,1,0,1]=3 ✓, f5 [0,1,0,1,1]=3 ✓, f6 [1,0,1,1,0]=3 ✓, f7 [0,1,1,0,0]=2 ✗
    assert confirmed == [False, False, False, False, True, True, True, False, False, False]


def test_small_object_filter_and_venom():
    class Clf(VenomClassifier):
        def predict(self, crop):
            return 0.8
    img = np.zeros((1000, 1000, 3), np.uint8)
    tiny, big = D([0, 0, 10, 10], 0, 0.9), D([100, 100, 600, 600], 0, 0.9)
    g = SnakeGuard(FakeDet([[tiny, big]] * 3), PipelineConfig(k_confirm=1), classifier=Clf())
    r = g.process(img)
    assert len(r.snakes) == 1 and r.snakes[0].p_venom == 0.8
    assert r.confirmed and r.venomous_alert and 0 < r.threat <= 100 and r.beep_interval_ms is not None
