"""End-to-end check of scripts/run_xai_bench.py: the CSV schema stays fixed across images with
different metric sets (polygon GT, FPs without GT), and resuming does not duplicate rows."""
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from snakeguard import PROJECT_ROOT, load_config

pytestmark = pytest.mark.gpu


def _run(tag, n):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_xai_bench.py"), "--split", "valid", "--n", str(n),
           "--tag", tag, "--methods", "eigencam", "random", "boxprior", "--expensive"]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, capture_output=True)


def test_bench_csv_schema_and_resume():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_xai_bench import COLUMNS
    tag = "pytest_schema"
    out = PROJECT_ROOT / load_config()["results_dir"] / "xai_bench" / f"valid_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    try:
        _run(tag, 25)
        df = pd.read_csv(out / "metrics.csv")
        assert list(df.columns) == COLUMNS
        d = df[df.method.notna()]
        assert d.ebpg_poly.notna().any(), "sample should contain polygon-labelled targets"
        assert d.method.isin(["eigencam", "random", "boxprior"]).all()
        # rows without GT (FPs) must have empty localization, not values shifted from other columns
        assert d[d.kind == "FP"].pg_box.isna().all()
        assert d[d.kind == "TP"].pg_box.notna().all()
        assert d.del_auc_norm.between(-0.01, 2).all() and d.score.between(0, 1).all()
        n_rows = len(df)
        _run(tag, 25)  # resume: nothing new to do
        assert len(pd.read_csv(out / "metrics.csv")) == n_rows
    finally:
        shutil.rmtree(out, ignore_errors=True)
