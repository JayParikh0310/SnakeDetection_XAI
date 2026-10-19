"""Dataset quality checks for the report (results/dataset_stats/quality.json):
1. Shortcut risk: how well does image resolution alone predict the class (snake vs look-alike file)?
2. Leakage: near-duplicate images across splits (64-bit difference hash, Hamming distance <= 4).
Usage: python scripts/dataset_quality.py
"""
import collections, json, os
from pathlib import Path
import numpy as np
from PIL import Image
from snakeguard import load_config

def dhash(im, n=8):
    g = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)

def main():
    cfg = load_config(); root = Path(cfg["data_root"])
    out = Path(cfg["results_dir"]) / "dataset_stats"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["train", "valid", "test"]:
        for f in os.listdir(root / split / "images"):
            with Image.open(root / split / "images" / f) as im:
                rows.append((split, "look-alike" if f.startswith("snake_like") else "snake", im.size, dhash(im), f))
    # 1. resolution -> class purity
    by_size = collections.defaultdict(collections.Counter)
    for s, c, size, _, _ in rows:
        by_size[size][c] += 1
    correct = sum(max(cnt.values()) for cnt in by_size.values())
    pure = sum(sum(cnt.values()) for cnt in by_size.values() if len(cnt) == 1)
    res = {"images": len(rows), "distinct_resolutions": len(by_size),
           "class_predictable_from_resolution": correct / len(rows),
           "images_with_single_class_resolution": pure / len(rows)}
    # 2. near-duplicates across splits (bucket by hash prefix chunks to keep it fast)
    from itertools import combinations
    hashes = [(r[3], r[0], r[4]) for r in rows]
    buckets = collections.defaultdict(list)
    for h, s, f in hashes:
        for k in range(4):  # 4 chunks of 16 bits: pigeonhole -> distance <= 3 shares a chunk
            buckets[(k, (h >> (16 * k)) & 0xFFFF)].append((h, s, f))
    pairs = set()
    for lst in buckets.values():
        if len(lst) > 400:  # degenerate (e.g. blank) hashes
            continue
        for (h1, s1, f1), (h2, s2, f2) in combinations(lst, 2):
            if s1 != s2 and bin(h1 ^ h2).count("1") <= 3:
                pairs.add(tuple(sorted([(s1, f1), (s2, f2)])))
    # verify candidates on pixels: the 64-bit hash alone matches different objects shot the same way
    # (e.g. belts on white backgrounds), so keep pairs whose 32x32 grayscale images correlate > 0.95
    def small(split, f):
        with Image.open(root / split / "images" / f) as im:
            a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        return (a - a.mean()) / (a.std() + 1e-6)
    cand = len(pairs)
    pairs = {p for p in pairs if float(np.mean(small(*p[0]) * small(*p[1]))) > 0.95}
    res["hash_candidates_before_pixel_check"] = cand
    cross = collections.Counter(tuple(sorted([a[0], b[0]])) for a, b in pairs)
    test_imgs_dup = {p[i][1] for p in pairs for i in (0, 1) if p[i][0] == "test"}
    res.update({"near_duplicate_pairs_across_splits": len(pairs),
                "pairs_by_split": {f"{a}-{b}": n for (a, b), n in cross.items()},
                "test_images_with_near_duplicate_elsewhere": len(test_imgs_dup),
                "test_share_with_near_duplicate": len(test_imgs_dup) / sum(r[0] == "test" for r in rows),
                "test_images_leaked": sorted(test_imgs_dup),
                "examples": [list(map(list, p)) for p in sorted(pairs)[:12]]})
    (out / "quality.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))

if __name__ == "__main__":
    main()
