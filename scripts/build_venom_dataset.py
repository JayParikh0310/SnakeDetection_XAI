"""B1: build the venom-classification dataset from SnakeCLEF 2023 (240 px images).

Step 1 (--plan, default): read the metadata + venom list, sample a capped subset per species
(whole observations), split train/valid/test by observation within each species, and write
`manifest.csv` + `manifest_summary.json`. No images are downloaded.
Step 2 (--extract): stream the train/val archives over HTTP and write only the manifest's images to
`<image_dir>/<image_path>`. Archives are never stored on disk. Already-extracted files are skipped,
and the stream is retried after network errors (the whole archive is re-read, extraction resumes).

Label: MIVS = medically important venomous snake (SnakeCLEF definition). Mildly venomous
rear-fanged species are 0.

Usage:
  python scripts/build_venom_dataset.py --plan
  python scripts/build_venom_dataset.py --extract
  python scripts/build_venom_dataset.py --mark      # refresh the `available` column only
Archives that were streamed to the end are recorded in `<root>/archives_done.json` and not re-streamed:
the 240 px train archive lacks 199 files that the metadata lists.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from snakeguard.venom.data import load_venom_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "valid", "test")


def read_metadata(sc: dict) -> pd.DataFrame:
    meta = Path(sc["meta_dir"])
    venom = pd.read_csv(meta / sc["venom_list"])
    parts = []
    for src, f in sc["metadata"].items():
        d = pd.read_csv(meta / f)[["observation_id", "binomial_name", "code", "image_path", "class_id"]]
        parts.append(d.assign(archive=src))
    df = pd.concat(parts, ignore_index=True).drop_duplicates()  # the iNat train metadata has exact duplicate rows
    if df.image_path.duplicated().any():
        raise ValueError("image_path is not unique after dropping exact duplicate rows")
    df = df.merge(venom, on="class_id", how="left", validate="many_to_one")
    if df.MIVS.isna().any():
        raise ValueError(f"{int(df.MIVS.isna().sum())} images have no venom status")
    df["venomous"] = df.pop("MIVS").astype(int)
    df["observation_id"] = df.observation_id.astype(str)
    return df


def sample_and_split(df: pd.DataFrame, sc: dict, seed: int) -> pd.DataFrame:
    """Per species: shuffle observations (seeded), take whole observations while the image count stays
    within the cap, then assign observations to splits in that shuffled order."""
    rng = np.random.default_rng(seed)
    fr = sc["split_frac"]
    obs_sizes = df.groupby(["class_id", "observation_id"]).size()
    venom = df.drop_duplicates("class_id").set_index("class_id").venomous
    chosen = []
    for cid, sizes in obs_sizes.groupby(level=0):
        cap = sc["cap_venomous"] if venom[cid] else sc["cap_harmless"]
        obs = sizes.index.get_level_values(1).to_numpy()[rng.permutation(len(sizes))]
        n_img, keep = 0, []
        for o in obs:
            k = int(sizes[(cid, o)])
            if n_img + k > cap and keep:   # always keep at least one observation per species
                continue
            keep.append(o); n_img += k
            if n_img >= cap:
                break
        n = len(keep)
        n_test = int(round(fr["test"] * n)) if n >= 3 else 0
        n_valid = int(round(fr["valid"] * n)) if n >= 3 else 0
        split = ["test"] * n_test + ["valid"] * n_valid + ["train"] * (n - n_test - n_valid)
        chosen += [(cid, o, s) for o, s in zip(keep, split)]
    sel = pd.DataFrame(chosen, columns=["class_id", "observation_id", "split"])
    out = df.merge(sel, on=["class_id", "observation_id"], how="inner")
    assert out.groupby("observation_id").split.nunique().max() == 1, "observation leaked across splits"
    return out.sort_values(["split", "class_id", "observation_id", "image_path"]).reset_index(drop=True)


def summarize(m: pd.DataFrame) -> dict:
    s = {"images": len(m), "observations": int(m.observation_id.nunique()), "species": int(m.class_id.nunique()),
         "species_venomous": int(m.drop_duplicates("class_id").venomous.sum())}
    for sp in SPLITS:
        g = m[m.split == sp]
        s[sp] = {"images": len(g), "venomous_images": int(g.venomous.sum()),
                 "venomous_frac": round(float(g.venomous.mean()), 3) if len(g) else None,
                 "observations": int(g.observation_id.nunique()), "species": int(g.class_id.nunique())}
    return s


def _member_to_rel(name: str) -> str | None:
    """'./SnakeCLEF2023-small_size/2021/X/1.jpg' -> '2021/X/1.jpg'. None for anything unexpected."""
    p = PurePosixPath(name[2:] if name.startswith("./") else name)
    parts = p.parts
    if len(parts) < 2 or parts[0] != "SnakeCLEF2023-small_size" or ".." in parts:
        return None
    return str(PurePosixPath(*parts[1:]))


def extract_archive(url: str, wanted: set[str], image_dir: Path, tries: int = 10) -> int:
    """Stream a .tar.gz and write only `wanted` regular image files. Returns the number written."""
    written = 0
    for attempt in range(1, tries + 1):
        todo = {w for w in wanted if not (image_dir / w).exists()}
        if not todo:
            return written
        print(f"{url.rsplit('/', 1)[-1]}: attempt {attempt}, {len(todo)} images to go", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for k, mem in enumerate(tar):
                    rel = _member_to_rel(mem.name)
                    if not mem.isreg() or rel is None or rel not in todo:
                        continue
                    if PurePosixPath(rel).suffix.lower() not in IMG_EXTS:
                        continue
                    dst = image_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    with tar.extractfile(mem) as src, open(tmp, "wb") as f:
                        f.write(src.read())
                    tmp.rename(dst)
                    written += 1
                    todo.discard(rel)
                    if written % 2000 == 0:
                        print(f"  {written} written, {len(todo)} to go", flush=True)
                    if not todo:
                        break
            return written
        except Exception as e:  # network drop mid-stream: retry, already-written files are skipped
            print(f"  stream failed ({type(e).__name__}: {e}); retrying in 30 s", flush=True)
            time.sleep(30)
    raise RuntimeError(f"gave up on {url} after {tries} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="download the manifest's images")
    ap.add_argument("--plan", action="store_true", help="(re)build the manifest only (default)")
    ap.add_argument("--mark", action="store_true", help="only refresh the manifest's `available` column")
    ap.add_argument("--force", action="store_true", help="re-stream archives already read to the end")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_venom_config(a.config)
    sc = cfg["snakeclef"]
    man = Path(sc["manifest"])
    image_dir = Path(sc["image_dir"])
    done_f = Path(sc["root"], "archives_done.json")  # archives streamed to the end (missing files are absent)

    if a.plan or not man.exists():
        m = sample_and_split(read_metadata(sc), sc, cfg["seed"])
        man.parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(man, index=False)
        summary = summarize(m)
        (man.parent / "manifest_summary.json").write_text(json.dumps({"config": sc, "seed": cfg["seed"],
                                                                      "summary": summary}, indent=1))
        print(json.dumps(summary, indent=1))
    if a.extract or a.mark:
        m = pd.read_csv(man, dtype={"observation_id": str})
        done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
        for src, arch in sc["archives"].items() if a.extract else ():
            if arch in done and not a.force:
                print(f"{arch}: already read to the end, skipped (--force to re-stream)", flush=True)
                continue
            wanted = set(m[m.archive == src].image_path)
            n = extract_archive(f"{sc['base_url']}/{arch}", wanted, image_dir)
            print(f"{arch}: {n} new images", flush=True)
            done.add(arch)
            done_f.write_text(json.dumps(sorted(done)))
        # Some metadata rows have no file in the 240 px archive. Keep the sampled manifest as is and
        # mark them, so loaders use `available == True`.
        m["available"] = [(image_dir / p).exists() for p in m.image_path]
        m.to_csv(man, index=False)
        missing = m.image_path[~m.available].tolist()
        print(f"done: {len(m) - len(missing)}/{len(m)} images present, {len(missing)} missing", flush=True)
        if missing:
            Path(sc["root"], "missing_images.txt").write_text("\n".join(missing))


if __name__ == "__main__":
    main()
