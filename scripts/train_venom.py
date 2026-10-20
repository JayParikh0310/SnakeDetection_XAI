"""B2: train + evaluate the venom classifier (VenomNet) on the SnakeCLEF 2023 subset.

Training is resumable per epoch (`last.pt`). The best epoch by valid AUROC is kept in `best.pt`. After
training, the operating threshold is chosen on valid (highest threshold with venomous recall >=
target_recall) and stored in `best.pt`; then valid + test are evaluated with it.

Outputs in results/venom_clf/<tag>/:
  run_config.json, history.csv, last.pt, best.pt (model + threshold + venomous species list)
  metrics.json             valid + test: AUROC, AP, recall/specificity/precision at the threshold and at 0.5,
                           95% CIs from a bootstrap over observations, species-head accuracy
  predictions_{valid,test}.csv  per image: p_venom (venom head), p_venom_species (species head), labels
  per_species_test.csv     per species: n, venomous, recall of the true label at the threshold

Usage:
  python scripts/train_venom.py --tag b0 [--epochs 12]
  python scripts/train_venom.py --tag bench --bench 30      # throughput only (30 train steps)
  python scripts/train_venom.py --tag b0 --eval-only        # re-evaluate best.pt
  python scripts/train_venom.py --tag b0_reg --set dropout=0.5 weight_decay=0.1 venom_label_smoothing=0.1 "crop_scale=[0.25, 1.0]"
  python scripts/train_venom.py --tag smoke --limit 256 --epochs 1 --n-boot 20   # end-to-end smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from snakeguard.venom import (VenomDataset, VenomNet, eval_transform, load_manifest, load_venom_config,
                              pick_device, train_transform)


def loaders(cfg, tr, limit=None):
    sc = cfg["snakeclef"]
    mans = {s: load_manifest(cfg, s) for s in ("train", "valid", "test")}
    if limit:  # smoke test: small seeded subset of every split
        mans = {s: m.sample(min(limit, len(m)), random_state=cfg["seed"]).reset_index(drop=True) for s, m in mans.items()}
    kw = dict(batch_size=tr["batch"], num_workers=tr["workers"], persistent_workers=tr["workers"] > 0)
    dl = {"train": DataLoader(VenomDataset(mans["train"], sc["image_dir"], train_transform(tr["img_size"], tr["crop_scale"])),
                              shuffle=True, drop_last=True, **kw)}
    for s in ("valid", "test"):
        dl[s] = DataLoader(VenomDataset(mans[s], sc["image_dir"], eval_transform(tr["img_size"])), shuffle=False, **kw)
    return mans, dl


@torch.no_grad()
def predict(model, dl, device, venom_mask):
    model.eval()
    venom_mask = venom_mask.to(device)
    pv, ps, sp_pred = [], [], []
    for x, _, _ in dl:
        v, s = model(x.to(device))
        prob_s = s.float().softmax(-1)
        pv.append(torch.sigmoid(v.float()).cpu()); ps.append((prob_s @ venom_mask).cpu())
        sp_pred.append(prob_s.argmax(-1).cpu())
    return torch.cat(pv).numpy(), torch.cat(ps).numpy(), torch.cat(sp_pred).numpy()


def pick_threshold(y, p, target_recall):
    """Highest threshold whose venomous recall is >= target (ties: keep all with p >= thr)."""
    pv = np.sort(p[y == 1])[::-1]
    k = int(math.ceil(target_recall * len(pv)))
    return float(pv[k - 1]) if k > 0 else 0.5


def binary_stats(y, p, thr):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {"threshold": thr, "recall_venomous": rec, "specificity": spec, "precision": tp / max(tp + fp, 1),
            "balanced_acc": (rec + spec) / 2, "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def bootstrap_ci(df, fn, n=1000, seed=0):
    """95% CI of fn(df) resampling whole observations (images of one observation are correlated)."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("observation_id").indices
    keys = np.array(list(groups))
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        vals.append(fn(df.iloc[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate(df, thr, n_boot):
    y, p = df.venomous.to_numpy(), df.p_venom.to_numpy()
    out = {"n": len(df), "venomous_frac": float(y.mean()),
           "auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
           "auroc_species_head": roc_auc_score(y, df.p_venom_species), "species_top1": float(df.species_correct.mean()),
           "at_threshold": binary_stats(y, p, thr), "at_0.5": binary_stats(y, p, 0.5)}
    if n_boot:
        out["auroc_ci"] = bootstrap_ci(df, lambda d: roc_auc_score(d.venomous, d.p_venom), n_boot)
        out["recall_ci"] = bootstrap_ci(df, lambda d: (d.p_venom[d.venomous == 1] >= thr).mean(), n_boot)
        out["specificity_ci"] = bootstrap_ci(df, lambda d: (d.p_venom[d.venomous == 0] < thr).mean(), n_boot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bench", type=int, default=0, help="only time this many train steps")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use N images per split")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override train settings, e.g. --set dropout=0.5 weight_decay=0.1 (values parsed as YAML)")
    a = ap.parse_args()
    cfg = load_venom_config(a.config)
    tr = cfg["train"]
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in tr:
            raise KeyError(f"unknown train setting {k}")
        tr[k] = yaml.safe_load(v)
    if a.epochs:
        tr["epochs"] = a.epochs
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = pick_device(tr["device"])
    out = Path(tr["out_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    mans, dl = loaders(cfg, tr, a.limit)
    allm = load_manifest(cfg)  # species index + venom status over the full manifest (subsets may lack species)
    n_species = int(allm.species_idx.max() + 1)
    sp_venom = allm.drop_duplicates("species_idx").set_index("species_idx").venomous
    venom_mask = torch.zeros(n_species); venom_mask[sp_venom.index[sp_venom == 1]] = 1
    model = VenomNet(n_species, tr["backbone"], pretrained=True, dropout=tr["dropout"]).to(device)
    print(f"device {device}, train {len(mans['train'])}, valid {len(mans['valid'])}, test {len(mans['test'])}, "
          f"species {n_species}", flush=True)

    steps_per_epoch = len(dl["train"])
    total = tr["epochs"] * steps_per_epoch
    warm = tr["warmup_epochs"] * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    start, best_auc, hist = 0, -1.0, []
    if (out / "last.pt").exists() and not a.bench:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_auc, hist = ck["epoch"] + 1, ck["best_auc"], ck["history"]
        print(f"resumed after epoch {ck['epoch']}", flush=True)
    else:
        (out / "run_config.json").write_text(json.dumps({"args": vars(a), "config": cfg, "torch": torch.__version__,
                                                         "device": str(device)}, indent=1))

    def save_best_meta(extra):
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        ck.update(extra)
        torch.save(ck, out / "best.pt")

    for ep in range(start, 0 if a.eval_only else tr["epochs"]):
        model.train()
        t0, n_seen, run = time.time(), 0, {"loss": 0.0, "venom": 0.0, "species": 0.0}
        for i, (x, yv, ys) in enumerate(dl["train"]):
            x, yv, ys = x.to(device), yv.to(device), ys.to(device)
            v, s = model(x)
            eps = tr["venom_label_smoothing"]
            lv = F.binary_cross_entropy_with_logits(v, yv * (1 - eps) + eps / 2)
            ls = F.cross_entropy(s, ys, label_smoothing=tr["label_smoothing"])
            loss = lv + tr["species_loss_weight"] * ls
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            n_seen += len(x)
            for k, val in (("loss", loss), ("venom", lv), ("species", ls)):
                run[k] += val.item() * len(x)
            if a.bench and i + 1 == a.bench:
                dt = time.time() - t0
                print(f"bench: {n_seen} imgs in {dt:.1f} s = {n_seen / dt:.0f} img/s "
                      f"-> ~{len(mans['train']) / (n_seen / dt) / 60:.1f} min/epoch (incl. warm-up)", flush=True)
                return
            if i % 100 == 0:
                print(f"ep {ep} step {i}/{steps_per_epoch} loss {run['loss'] / n_seen:.4f} "
                      f"{n_seen / (time.time() - t0):.0f} img/s", flush=True)
        pv, ps, spp = predict(model, dl["valid"], device, venom_mask)
        mv = mans["valid"]
        auc = roc_auc_score(mv.venomous, pv)
        h = {"epoch": ep, **{k: v / n_seen for k, v in run.items()}, "valid_auroc": auc,
             "valid_auroc_species_head": roc_auc_score(mv.venomous, ps),
             "valid_species_top1": float((spp == mv.species_idx.to_numpy()).mean()),
             "lr": sched.get_last_lr()[0], "minutes": (time.time() - t0) / 60}
        hist.append(h)
        print(json.dumps(h), flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if auc > best_auc:
            best_auc = auc
            torch.save({"model": model.state_dict(), "n_species": n_species, "backbone": tr["backbone"],
                        "img_size": tr["img_size"], "epoch": ep, "valid_auroc": auc,
                        "venomous_species": venom_mask.nonzero()[:, 0].tolist()}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_auc": best_auc, "history": hist}, out / "last.pt")

    # ---- final evaluation with the best epoch
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    preds = {}
    for s in ("valid", "test"):
        pv, ps, spp = predict(model, dl[s], device, venom_mask)
        d = mans[s][["image_path", "observation_id", "class_id", "binomial_name", "code", "venomous", "species_idx"]].copy()
        d["p_venom"], d["p_venom_species"], d["species_correct"] = pv, ps, spp == d.species_idx.to_numpy()
        d.to_csv(out / f"predictions_{s}.csv", index=False)
        preds[s] = d
    thr = pick_threshold(preds["valid"].venomous.to_numpy(), preds["valid"].p_venom.to_numpy(), tr["target_recall"])
    save_best_meta({"threshold": thr, "target_recall": tr["target_recall"]})
    metrics = {"best_epoch": int(ck["epoch"]), "threshold": thr, "target_recall": tr["target_recall"],
               **{s: evaluate(preds[s], thr, a.n_boot if s == "test" else 0) for s in ("valid", "test")}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    t = preds["test"]
    per = t.assign(correct=(t.p_venom >= thr) == (t.venomous == 1)).groupby(["class_id", "binomial_name"]).agg(
        n=("correct", "size"), venomous=("venomous", "first"), acc=("correct", "mean"),
        mean_p=("p_venom", "mean")).reset_index().sort_values(["venomous", "acc"], ascending=[False, True])
    per.to_csv(out / "per_species_test.csv", index=False)
    print(json.dumps(metrics, indent=1), flush=True)


if __name__ == "__main__":
    main()
