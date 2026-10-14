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

# Switches/Thresholds
RULE1_TSH = 0.6
RULE2_TSH = 0.5
RULE3_TSH = 0.65  # original: 0.4
SIZE_TOL = 2      # tolerance for size 
DEBUG = False 

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


def rule1_metadata(target, input_path):
    """File size + dimensions. Compression and crops leave size fingerprints."""
    out = {"rule": 1, "name": "Metadata", "fired": False, "score": 0,
           "out_of": 30, "note": "Size ratio 0.00", "metric": 0.0}
    try:
        src_size = os.path.getsize(target["path"])
        in_size = os.path.getsize(input_path)
        src_w, src_h = _size(target["path"])
        in_w, in_h = _size(input_path)
        size_ratio = min(src_size, in_size) / max(src_size, in_size)
        area_kept = (in_w * in_h) / max(1, src_w * src_h)

        out["note"] = f"Size ratio {size_ratio:.2f}"

        metric = 0.5 * size_ratio + 0.5 * area_kept

        # Format change of same picture
        src_fmt, in_fmt = _fmt(target["path"]), _fmt(input_path)
        if (in_w, in_h) == (src_w, src_h) and src_fmt != in_fmt and "?" not in (src_fmt, in_fmt):
          metric = max(metric, 0.95)
          out["note"] = f"Format change {src_fmt} to {in_fmt}, size_ratio {size_ratio:.2f}"

        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        if out["metric"] >= RULE1_TSH:
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

        out["metric"] = round(max(0.0, min(1.0, corr)), 3)
        out["note"] = f"Correlation {out['metric']:.2f}"

        if out["metric"] >= RULE2_TSH:
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
      src_g = _gray(target["path"])
      nd_g = _gray(input_path)
      if src_g is None or nd_g is None:
        return out
          
      src_g = _downscale(src_g, 256)
      nd_g = _downscale(nd_g, 256)

      src_w, src_h = _size(target["path"]) or (0,0)
      in_w, in_h = _size(input_path) or(0,0)
      if min(src_w, src_h, in_w, in_h) > 0:
        s = (src_g.shape[1]/src_w) / (nd_g.shape[1]/in_w)
        nd_g = cv2.resize(nd_g, (max(1, int(round(nd_g.shape[1] * s))), 
        max(1, int(round(nd_g.shape[0] * s)))),
        interpolation = cv2.INTER_AREA)

      if (src_g.shape[0] + SIZE_TOL < nd_g.shape[0]) or (src_g.shape[1] + SIZE_TOL < nd_g.shape[1]):
          return out 

      nd_g = nd_g[:src_g.shape[0], :src_g.shape[1]]

      res = cv2.matchTemplate(src_g, nd_g, cv2.TM_CCOEFF_NORMED)
      metric = float(np.nan_to_num(res.max()))

      out["metric"] = round(max(0.0, min(1.0, metric)), 3)
      out["note"] = f"Match score {out['metric']:.2f}"
      if out["metric"] >= RULE3_TSH:
          out["fired"] = True
          out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
      if DEBUG:
        raise
    return out