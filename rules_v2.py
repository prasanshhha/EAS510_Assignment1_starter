"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Rules 1-3 are kept exactly as in `rules.py`. Your job is to ADD Rule 4 that
targets the systematic weakness you diagnosed from results_v1_hard.txt.

Constraint: the rules' `out_of` weights must still sum to 100 so the Final
Score stays on a /100 scale (the format validator enforces "Final Score:
<s>/100" equals the sum of the block's rule scores).

Example split that keeps the original spirit: Rules 1-3 -> 25/25/40 and
Rule 4 -> 10 (or trim the losers more aggressively).
"""

import cv2
import numpy as np
import rules

#: Import the original rules and append your new one.
RULES = (
    "rule1_metadata_v2",
    "rule2_histogram_v2",
    "rule3_template_v2",
    "rule4_edges",
  )

# Settings
RULE4_TSH = 0.5
RULE1_POINTS = 15
RULE2_POINTS = 25
RULE3_POINTS = 30
RULE4_POINTS = 30

def _rescale_rule(result, new_out_of):
    # Adjusting score to new point budget while keeping rule metric and thresholds
    # result = result.copy()
    result["out_of"] = new_out_of
    if result["fired"]:
      result["score"] = int(round(new_out_of * result["metric"]))
    else:
      result["score"] = 0.0

    return result

def rule1_metadata_v2(target, input_path):
    result = rules.rule1_metadata(target, input_path)
    return _rescale_rule(result, RULE1_POINTS)

def rule2_histogram_v2(target, input_path):
    result = rules.rule2_histogram(target, input_path)
    return _rescale_rule(result, RULE2_POINTS)

def rule3_template_v2(target, input_path):
    result = rules.rule3_template(target, input_path)
    return _rescale_rule(result, RULE3_POINTS)

def rule4_edges(target, input_path):
    # Matching local visual features
    out = {
        "rule": 4,
        "name": "ORB Keypoints",    #Edges
        "fired": False,
        "score": 0,
        "out_of": RULE4_POINTS,
        "note": "Not implemented",
        "metric": 0.0,
    }

    try:
      src_g = rules._gray(target["path"])
      in_g = rules._gray(input_path)

      if src_g is None or in_g is None:
        return out
      
      orb = cv2.ORB_create(nfeatures=1500)

      src_kp, src_des = orb.detectAndCompute(src_g, None)
      in_kp, in_des = orb.detectAndCompute(in_g, None)

      if src_des is None or in_des is None:
        out["note"] = "No usable ORB descriptors"
        return out
      
      if len(src_kp) < 4 or len(in_kp) < 4:
        out["note"] = "Fewer than 4 keypoints"
        return out
      
      matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

      pairs = matcher.knnMatch(src_des, in_des, k=2)

      good = [
        m for pair in pairs
        if len(pair) == 2
        for m,n in [pair]
        if m.distance < 0.75 * n.distance
      ]

      if len(good) < 4:
        out["note"] = f"Good matches {len(good)}"
        return out

      # Coordinates of corresponding points
      src_pts = np.float32([
        src_kp[m.queryIdx].pt for m in good
      ]).reshape(-1,1,2)

      in_pts = np.float32([
        in_kp[m.trainIdx].pt for m in good
      ]).reshape(-1,1,2)

      # Find geometrically consistent matches
      _, mask = cv2.findHomography(
        src_pts, in_pts, cv2.RANSAC, 5.0
      )

      if mask is None:
        out["note"] = "Homography estimation failed"
        return out
      
      inliers = int(mask.ravel().sum())

      # Reward both number and proportion of geometrically consistent feature matches
      inlier_ratio = inliers / len(good)
      count_score = min(inliers / 12.0, 1.0)
      metric = inlier_ratio * count_score

      out["metric"] = round(float(metric), 3)
      out["note"] = f"Match Score {metric:.2f}"

      if metric >= RULE4_TSH:
        out["fired"] = True
        out["score"] = int(round(RULE4_POINTS * metric))
      
    except Exception:
      if rules.DEBUG:
        raise
    
    return out
      
