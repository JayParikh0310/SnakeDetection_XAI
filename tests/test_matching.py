import numpy as np

from snakeguard.data import GTObject
from snakeguard.detection.yolo import Detection
from snakeguard.eval.matching import match


def D(box, cls, score):
    b = np.array(box, float)
    return Detection(b, b, cls, score, 0)


def test_match_kinds():
    gts = [GTObject(0, np.array([0, 0, 10, 10.])), GTObject(1, np.array([50, 50, 60, 60.])),
           GTObject(0, np.array([100, 100, 110, 110.]))]
    dets = [D([0, 0, 10, 10], 0, 0.9),       # TP
            D([50, 50, 60, 61], 0, 0.8),     # box ok, wrong class -> confusion
            D([200, 200, 210, 210], 1, 0.7),  # FP
            D([0, 0, 10, 11], 0, 0.6)]       # duplicate of first -> FP
    kinds = sorted(m.kind for m in match(dets, gts))
    assert kinds == sorted(["TP", "confusion", "FP", "FP", "FN"])


def test_match_empty():
    assert match([], []) == []
    assert [m.kind for m in match([], [GTObject(0, np.array([0, 0, 1, 1.]))])] == ["FN"]
    assert [m.kind for m in match([D([0, 0, 1, 1], 0, .5)], [])] == ["FP"]
