"""EAS 510 - Project 1: Rule functions for the Digital Forensics Apprentice.

Each rule compares one suspect image against one registered original and
returns a dict:

    {
        "rule": 1,
        "name": "Metadata",
        "fired": bool,          # True if the rule found supporting evidence
        "score": int,           # points awarded out of "out_of"
        "out_of": int,
        "note": "Size ratio 0.85",   # short human-readable metric
        "metric": 0.85,              # raw similarity measure in [0, 1]
    }

The points budgets are fixed by the assignment: Rule 1 = 30, Rule 2 = 30,
Rule 3 = 40. The detector sums them into a 0-100 confidence score.

These starter implementations are deliberately WEAK baselines (they hinge on
simple thresholds). Improving them is the assignment.

Reads are cached per path so one suspect against many originals does not
re-decode images needlessly.
"""

import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image

#: Rule names used by SimpleDetector.evaluate() (V1).
RULES = ("rule1_metadata", "rule2_histogram", "rule3_template")

DEBUG = False 
# Switches/Thresholds
RULE1_TSH = 0.6           # original: 0.6
RULE2_TSH = 0.5           # original: 0.5
RULE3_TSH = 0.65          # original: 0.4
PIXEL_CONFIRM = 0.65
MAX_DENSITY_GROWTH = 1.5  # edited copy cannot have density >1.5* bytes per pixel
MAX_DETAIL_GROWTH = 3.0   # edited copy cannot be >3* busier than original
SIZE_TOL = 2              # tolerance for size 

@lru_cache(maxsize=256)
def _arr(path):
    return cv2.imread(path)


def _downscale(img, max_dim=512):
    """Fit an image into `max_dim` before expensive cv2 work."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


@lru_cache(maxsize=256)
def _hist(path):
    img = _arr(path)
    if img is None:
        return None
    img = _downscale(img, 128)
    hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
    hist_all = np.concatenate(hists).ravel().astype(np.float32)
    cv2.normalize(hist_all, hist_all)
    return hist_all


@lru_cache(maxsize=256)
def _hist_hs(path):
    # hue/saturation histogram that ignores brightness
    img = _arr(path)
    if img is None:
      return None
    hsv = cv2.cvtColor(_downscale(img, 128), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1], None, [16,8], [0, 180, 0, 256]).ravel()
    h = h.astype(np.float32)
    cv2.normalize(h,h)
    return h


@lru_cache(maxsize=256)
def _gray(path):
    img = _arr(path)
    if img is None:
        return None
    return _downscale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


@lru_cache(maxsize=256)
def _size(path):
    # replaced PIL with cv2 decoder
    img = _arr(path)
    if img is None:
      return None
    h, w = img.shape[:2]
    return (w,h)


@lru_cache(maxsize=256)
def _fmt(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        return "?"
    if head[:2] == b"\xff\xd8":
        return "JPEG"
    if head[:4] == b"\x89PNG":
        return "PNG"
    return "?"


@lru_cache(maxsize=256)
def _detail(path):
    # pixel to pixel busyness
    img = _arr(path)
    if img is None:
      return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    y0, x0 = max(0, h // 2 - 128), max(0, w // 2 - 128)
    p = g[y0:y0 + 256, x0:x0 + 256]
    if p.shape[0] < 2 or p.shape[1] < 2:
      return 0.0
    return float(np.abs(np.diff(p, axis=0)).mean() + np.abs(np.diff(p, axis=1)).mean()) / 2


@lru_cache(maxsize=256)
def _template_score(target_path, input_path):
    src_g = _gray(target_path)
    nd_g = _gray(input_path)
    if src_g is None or nd_g is None:
        return 0.0
        
    src_g = _downscale(src_g, 256)
    nd_g = _downscale(nd_g, 256)

    src_w, src_h = _size(target_path) or (0,0)
    in_w, in_h = _size(input_path) or(0,0)
    if min(src_w, src_h, in_w, in_h) > 0:
      s = (src_g.shape[1]/src_w) / (nd_g.shape[1]/in_w)
      nd_g = cv2.resize(nd_g, (max(1, int(round(nd_g.shape[1] * s))), 
      max(1, int(round(nd_g.shape[0] * s)))),
      interpolation = cv2.INTER_AREA)

    if (src_g.shape[0] + SIZE_TOL < nd_g.shape[0]) or (src_g.shape[1] + SIZE_TOL < nd_g.shape[1]):
        return 0.0 

    nd_g = nd_g[:src_g.shape[0], :src_g.shape[1]]
    if nd_g.std() < 1e-3:
      return 0.0

    res = cv2.matchTemplate(src_g, nd_g, cv2.TM_CCOEFF_NORMED)
    metric = float(np.nan_to_num(res.max()))
    
    return metric

def rule1_metadata(target, input_path):
    """File size + dimensions. Compression and crops leave size fingerprints."""
    out = {"rule": 1, "name": "Metadata", "fired": False, "score": 0,
           "out_of": 30, "note": "Size ratio 0.00", "metric": 0.0}
    try:
        src_size = os.path.getsize(target["path"])
        in_size = os.path.getsize(input_path)
        src_w, src_h = _size(target["path"]) or (0, 0)
        in_w, in_h = _size(input_path) or (0, 0)
        size_ratio = min(src_size, in_size) / max(src_size, in_size)
        area_kept = (in_w * in_h) / max(1, src_w * src_h)

        out["note"] = f"Size ratio {size_ratio:.2f}"

        # bytes per pixel
        src_bpp = src_size / (src_w * src_h)
        in_bpp = in_size / (in_w * in_h)
        growth = in_bpp / src_bpp
        if growth <= MAX_DENSITY_GROWTH:
          density = 1.0
        elif growth >= 4.0:
          density = 0.0
        else:
          density = (4.0 - growth) / (4.0 - MAX_DENSITY_GROWTH)
        
        # Dimension score
        if (in_w, in_h) == (src_w, src_h):
          dim_score = 1.0
        elif in_w <= src_w + SIZE_TOL and in_h <= src_h + SIZE_TOL and area_kept >= 0.05:
          dim_score = 0.8
        else:
          dim_score = 0.0
        metric = dim_score * density

        # Format change of same picture
        src_fmt, in_fmt = _fmt(target["path"]), _fmt(input_path)
        if (in_w, in_h) == (src_w, src_h) and src_fmt != in_fmt and "?" not in (src_fmt, in_fmt):
          metric = max(metric, 0.95)
          out["note"] = f"Format change {src_fmt} to {in_fmt}, size_ratio {size_ratio:.2f}"

        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        if out["metric"] >= RULE1_TSH and _template_score(target["path"], input_path) >= PIXEL_CONFIRM:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
      if DEBUG:
        raise
    return out


def rule2_histogram(target, input_path):
    """Color histogram correlation. Robust to crops and spatial changes."""
    out = {"rule": 2, "name": "Histogram", "fired": False, "score": 0,
           "out_of": 30, "note": "Correlation 0.00", "metric": 0.0}
    try:
        hs, hi = _hist(target["path"]), _hist(input_path)
        if hs is None or hi is None:
            return out
        corr = float(np.nan_to_num(cv2.compareHist(hs, hi, cv2.HISTCMP_CORREL)))

        # Hue/Saturation hist ignoring brightness
        hs2, hi2 = _hist_hs(target["path"]), _hist_hs(input_path)
        if hs2 is not None and hi2 is not None:
          corr = max(corr, float(np.nan_to_num(cv2.compareHist(hs2, hi2, cv2.HISTCMP_CORREL))))

        out["metric"] = round(max(0.0, min(1.0, corr)), 3)
        out["note"] = f"Correlation {out['metric']:.2f}"

        # Detail growth
        d_src, d_in = _detail(target["path"]), _detail(input_path)
        too_busy = d_src is not None and d_in is not None and d_in > MAX_DETAIL_GROWTH * d_src + 2.0

        if out["metric"] >= RULE2_TSH and not too_busy and _template_score(target["path"], input_path) >= PIXEL_CONFIRM:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
      if DEBUG:
        raise
    return out


def rule3_template(target, input_path):
    """Template matching: Detects when the suspect is contained in the original.

    cv2.matchTemplate needs the template (suspect) smaller than the target
    (original); on rotated/resized suspects this rule degrades fast. Fixing
    that becomes Rule 4 territory.
    """
    out = {"rule": 3, "name": "Template", "fired": False, "score": 0,
           "out_of": 40, "note": "Match score 0.00", "metric": 0.0}
    try:
        metric = _template_score(target["path"], input_path)

        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        out["note"] = f"Match score {out['metric']:.2f}"
        if out["metric"] >= RULE3_TSH:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
      if DEBUG:
        raise
    return out