# EAS 510 - Assignment 1: Digital Forensics Apprentice

A rule-based expert system that matches modified images back to their originals.

This is the **starter repository** for Project 1. Two repositories matter:

| Repository | Role |
|------------|------|
| `delveccj/EAS510_Assignment1_starter` (this repo) | Your **code** lives here. You will fork it, work in your fork, and push your final submission here. |
| `delveccj/EAS510_Assignment1` | The **dataset** (read-only). Clone it for the images; never commit it to your fork. |

## Setup on your Codio box

```bash
# 1. Fork this repo on GitHub, then in your box:
git clone https://github.com/YOUR-USERNAME/EAS510_Assignment1_starter.git
cd EAS510_Assignment1_starter

# 2. Clone the read-only dataset as a sibling folder (images live there):
cd ~/workspace
git clone https://github.com/delveccj/EAS510_Assignment1.git

# 3. Install dependencies:
cd ~/workspace/EAS510_Assignment1_starter
python3 -m pip install -r requirements.txt

# 4. Point this box at YOUR fork (run once):
./setup_git.sh
```

## Repository layout

```
forensics_detective.py   SimpleDetector: register targets + find_best_match
rules.py                 Rule functions (Rule 1: Metadata, Rule 2: Histogram, Rule 3: Template)
test_system.py           Runs the detector over the data folders and writes results_*.txt
scripts/check_output_format.py   Validates a results file against the required format
setup_git.sh, submit.sh  One-time setup + commit/push helper ("backup button")
```

## The task in one paragraph

Implement three interpretable rules (metadata, color histogram, template matching)
that combine into a 0-100 confidence score. Run your system on `modified_images/`
and `random/` and save the full output as `results_v1.txt`. In Phase 2, run on
`hard/`, diagnose a systematic failure, add a **Rule 4**, and save
`results_v1_hard.txt` and `results_v2.txt`. Full instructions are in the Codio guide
and in the assignment PDF in UBLearns.

## Running

```bash
# Phase 1 (easy + random) -> results_v1.txt
python3 test_system.py --modified --random --output results_v1.txt

# Phase 1 on hard cases -> results_v1_hard.txt
python3 test_system.py --hard --output results_v1_hard.txt

# Phase 2 (everything) -> results_v2.txt
python3 test_system.py --modified --hard --random --output results_v2.txt

# Validate any results file against the required format:
python3 scripts/check_output_format.py results_v1.txt
```

The data folders are resolved from a sibling clone of `EAS510_Assignment1`.
If your data is elsewhere, pass `--data-dir /path/to/EAS510_Assignment1`.

## Output format (do not alter)

```
Processing: modified_image_01.jpg
Rule 1 (Metadata): FIRED - Size ratio 0.85 -> 20/30 points
Rule 2 (Histogram): FIRED - Correlation 0.92 -> 25/30 points
Rule 3 (Template): FIRED - Match score 0.76 -> 30/40 points
Final Score: 75/100 -> MATCH to original_03.jpg
```

## Backup doctrine

Your Codio box can be reset at any time. **Your fork on GitHub is the only safe
copy.** After every milestone run `./submit.sh "describe what you did"`. A box
restart never touches GitHub.

## License

Apache 2.0. See `LICENSE`.

## Observed weakness in V1: what failed and why
V1 used metadata, histogram, and template matching, but struggled with combined transformations such as cropping, resizing, and compression. 
For example, `original_08__crop_keep60__bright__compress__q55__v2.jpg` was rejected with a score of 38/100 because Rules 1 and 2 failed to provide 
sufficient evidence. Rule 3 assumes the suspect image remains at approximately the same scale and alignment as the original. It fired on all 30 
images of variants v1, v2, and v5, but only 2/30 images of v3, v4, and v6. When Rule 3 contributed zero, Rules 1 and 2 could contribute at most 
60 points combined, leaving 27/30 images scoring 43–55/100, around the 50-point cutoff. V1 correctly identified 43/60 hard images, incorrectly 
matched four, and rejected 13. For example, `original_04__resize_scale85__compress__q60__v3.jpg` was incorrectly matched to original_03 with a score of 52/100.

## Design decision for V2: what Rule 4 is and why you chose it
V2 adds Rule 4 using ORB keypoint matching and RANSAC to identify geometrically consistent features between images. It detects up to 1,500 keypoints per image, 
applies the 0.75 ratio test to descriptor matches, and uses RANSAC to estimate a geometric transformation. The metric is inlier_ratio × min(inliers / 12, 1), 
with a firing threshold of 0.5. For example, `original_00__resize_scale114__compress__q45__v3.jpg` produced 722 inliers from 732 good matches, despite Rule 3 scoring 
only 0.44. The weights are 15, 25, 30, and 30 for Rules 1–4, respectively. Rule 1's weight was reduced because it did not fire on 20 modified images involving 25% and 50% crops.

## Effect of the change: accuracy before/after on easy vs hard
On the hard set of 60 images, V1 correctly identified 43, incorrectly matched four, and rejected 13. V2 correctly identified 56, made no incorrect 
matches, and rejected four, increasing the correct identification rate from 71.7% to 93.3%. Across the modified and hard sets, correct identifications 
increased from 102/120 to 110/120. V2 correctly matched 54/60 modified images and rejected all 15 random images. Rule 4 fired on 59/60 hard images, 
49/60 modified images, and 0/15 random images; firing indicates that the rule met its own threshold, not necessarily that the final identification was correct.

## Trade-offs: what new costs or risks did Rule 4 introduce
ORB and RANSAC add computation compared with simple metadata checks. Matching can also fail when images have few distinctive features or undergo severe cropping 
and compression. In the tests, Rule 4 did not fire on any of the ten 25% crop images, which had only 4–8 inliers. Reducing Rule 3's weight from 40 to 30 also 
removed some score margin, causing five crops to be rejected, including `modified_00_crop_25pct.jpg`. This illustrates the trade-off between improving identification 
of transformed images and preserving enough score for difficult genuine matches.