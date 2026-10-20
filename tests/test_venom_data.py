"""B1: venom dataset builder (no downloads, synthetic metadata)."""
import sys

import numpy as np
import pandas as pd

from snakeguard import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_venom_dataset import _member_to_rel, sample_and_split  # noqa: E402

SC = {"cap_venomous": 10, "cap_harmless": 4, "split_frac": {"train": 0.7, "valid": 0.15, "test": 0.15}}


def _meta(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(20):
        for o in range(rng.integers(1, 15)):
            for i in range(rng.integers(1, 4)):  # 1-3 photos per observation
                rows.append({"class_id": cid, "observation_id": f"{cid}_{o}", "image_path": f"{cid}/{o}_{i}.jpg",
                             "venomous": int(cid % 4 == 0), "archive": "train"})
    return pd.DataFrame(rows)


def test_split_by_observation_and_caps():
    df = _meta()
    m = sample_and_split(df, SC, seed=42)
    assert m.groupby("observation_id").split.nunique().max() == 1   # no observation in two splits
    assert set(m.split) <= {"train", "valid", "test"}
    assert m.class_id.nunique() == df.class_id.nunique()              # every species kept
    per = m.groupby("class_id").agg(n=("image_path", "size"), v=("venomous", "first"), obs=("observation_id", "nunique"))
    # cap holds unless the species' single kept observation is itself larger than the cap
    over = per[per.n > np.where(per.v == 1, SC["cap_venomous"], SC["cap_harmless"])]
    assert (over.obs == 1).all()
    # whole observations are kept
    kept = m.groupby("observation_id").size()
    full = df.groupby("observation_id").size()
    assert (kept == full[kept.index]).all()


def test_sampling_is_deterministic():
    df = _meta()
    a, b = sample_and_split(df, SC, seed=42), sample_and_split(df, SC, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_member_names():
    assert _member_to_rel("./SnakeCLEF2023-small_size/2021/A_b/1.jpg") == "2021/A_b/1.jpg"
    assert _member_to_rel("SnakeCLEF2023-small_size/2021/A_b/1.jpeg") == "2021/A_b/1.jpeg"
    assert _member_to_rel("SnakeCLEF2023-small_size/../../etc/passwd") is None
    assert _member_to_rel("other/2021/x.jpg") is None
    assert _member_to_rel("SnakeCLEF2023-small_size") is None
