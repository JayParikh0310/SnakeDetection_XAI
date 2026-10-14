from .base import DetectorExplainer, Explainer, normalize
from .cam import EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
from .perturbation import DRISE, BoxPrior, Occlusion, RandomMap


def build_explainers(detector, cfg: dict | None = None, names=None) -> dict[str, DetectorExplainer]:
    """All detector explainers used in the benchmark, keyed by name.
    Core (from the proposal): eigencam, gradcampp, occlusion. The rest are extras, ablations and baselines."""
    cfg = cfg or {}
    occ = cfg.get("occlusion", {})
    dr = cfg.get("drise", {})
    seed = cfg.get("seed", 42)
    factories = {
        "eigencam": lambda: EigenCAM(detector),
        "eigencam_p5": lambda: EigenCAM(detector, scales=("P5",), name="eigencam_p5"),
        "eigencam_centered": lambda: EigenCAM(detector, center=True, name="eigencam_centered"),
        "gradcam": lambda: GradCAM(detector),
        "gradcampp": lambda: GradCAMPlusPlus(detector),
        "gradcampp_p5": lambda: GradCAMPlusPlus(detector, scales=("P5",), name="gradcampp_p5"),
        "gradcampp_p3": lambda: GradCAMPlusPlus(detector, scales=("P3",), name="gradcampp_p3"),
        "gradcampp_p4": lambda: GradCAMPlusPlus(detector, scales=("P4",), name="gradcampp_p4"),
        "gradcampp_fmax": lambda: GradCAMPlusPlus(detector, fusion="max", name="gradcampp_fmax"),
        "gradcampp_fsum": lambda: GradCAMPlusPlus(detector, fusion="sum", name="gradcampp_fsum"),
        "eigencam_fsum": lambda: EigenCAM(detector, fusion="sum", name="eigencam_fsum"),
        "gradcampp_contrast": lambda: GradCAMPlusPlus(detector, mode="contrastive", name="gradcampp_contrast"),
        "layercam": lambda: LayerCAM(detector),
        "layercam_fsum": lambda: LayerCAM(detector, fusion="sum", name="layercam_fsum"),
        "layercam_p5": lambda: LayerCAM(detector, scales=("P5",), name="layercam_p5"),
        "occlusion": lambda: Occlusion(detector, **occ),
        "drise": lambda: DRISE(detector, seed=seed, **dr),
        "random": lambda: RandomMap(detector, seed=seed),
        "boxprior": lambda: BoxPrior(detector),
    }
    names = names or list(factories)
    unknown = set(names) - set(factories)
    if unknown:
        raise KeyError(f"unknown explainers: {unknown}")
    return {n: factories[n]() for n in names}


__all__ = ["Explainer", "DetectorExplainer", "normalize", "EigenCAM", "GradCAM", "GradCAMPlusPlus", "LayerCAM",
           "Occlusion", "DRISE", "RandomMap", "BoxPrior", "build_explainers"]
