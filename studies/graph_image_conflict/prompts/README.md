# Prompt templates

Draft for review; part of the plan once frozen (`plan-v1`). Every call records the hash of the assembled prompt (D18), so any change to these files after freezing is visible.

**Source:** adapted from LightEMMA (`predict.py`, MIT licence), github.com/michigan-traffic-lab/LightEMMA, commit `4906a89b7804f82da8fd3b1ac36452d41c51b1d0` (2025-08-23). Wording is kept as close to LightEMMA's as the design allows; every change is listed below.

## Files

| File | Component | Placeholders |
| --- | --- | --- |
| `01_role.txt` | Role line (LightEMMA, unchanged; to be settled by the role-line comparison in D19, step 3) | none |
| `02_camera.txt` | Camera line (LightEMMA, unchanged) | none |
| `03_ego_history.txt` | Past 3 s of speed and curvature (LightEMMA) | `{speeds}`, `{curvatures}` |
| `04_scene_graph.txt` | Scene graph (new) | `{scene_graph}`: the graph text from D8, one object per line, or `No objects detected in view.` |
| `05_task.txt` | Reasoning steps: LightEMMA's three stages, as steps of one call | none |
| `06_output_format.txt` | Output instruction (LightEMMA, adapted for a reasoning model) | none |

## Assembly

- **One user message** per call. Its content is the image part first (if the condition has one), then one text part.
- **Text part:** the condition's components, in file order, joined with a blank line (`\n\n`).
- **Reasoning** is switched on through Gemma 4's own chat-template mechanism (D1), not through the prompt text. The final answer is parsed from the answer, never from the reasoning.
- **Image:** the real CAM_FRONT JPEG or the gray image (D17), at the 1120-token budget (D1).

| Condition | Image part | Components |
| --- | --- | --- |
| Image only | real | 01, 02, 03, 05, 06 |
| Ego history only | none | 01, 03, 05, 06 |
| Image + correct / broken graph | real | 01, 02, 03, 04, 05, 06 |
| Gray + correct / broken graph | gray | 01, 02, 03, 04, 05, 06 (identical text to the real-image condition) |
| Graph only, correct / broken | none | 01, 03, 04, 05, 06 |

Rules:

- **No graph** (image only, ego history only): component 04 is left out entirely. It is never replaced by `No objects detected in view.`, which would claim the road is empty.
- **No image** (ego history only, graph only): component 02 is left out, and no image part is sent.
- **Gray image:** the text is byte-for-byte the same as with the real image, so the only difference is the picture.
- **The graph is introduced neutrally** ("The perception system detected..."): no warning that it may be wrong, and no claim that it is accurate.

## History values (from LightEMMA's code)

- 8 ego poses, from 7 keyframes before the current one up to the current one, taken from the CAM_FRONT sample's ego pose and transformed into the current ego frame.
- **Speeds:** 6 values, distance between consecutive poses divided by their time difference; values below 0.2 m/s are set to 0.
- **Curvatures:** 6 values, from each pose and its two neighbours (circle through three points, signed, positive = left); values beyond +/-0.2 per metre are set to 0.
- Rounded to 3 decimals and written as Python lists, e.g. `[8.105, 8.212, 8.3, 8.341, 8.402, 8.45]`.

## Changes from LightEMMA

| LightEMMA | Here | Why |
| --- | --- | --- |
| Three calls: scene description, then intent, then waypoints, each with the image | One call; the three stages are reasoning steps (`05_task.txt`) | Gemma 4 reasons natively (D1); one call per repeat keeps cost at a third; the scene graph is then available at every stage |
| No scene-graph input | `04_scene_graph.txt` added between the history and the task | The variable under study |
| Camera line always present | Left out when there is no image | The graph-only and ego-only conditions have no image |
| "...resolution is[8.1, ...]" (missing space) | Space added | Formatting fix |
| "ONLY return the answers in the required format, do not include punctuation or text." | The final answer must contain only the list, after the reasoning | Reasoning comes first; the old wording contradicts the list's own punctuation |
| Units implicit | Speed in m/s, curvature in 1/m, positive = left, stated in the output instruction | Removes ambiguity in the answer |

## Findings from the LightEMMA check (carried into the plan on 2026-11-27: D3, D12, D13)

1. **History length (D3):** LightEMMA needs **7** keyframes before the current one (8 poses: 6 speeds and 6 curvatures, each curvature needing a neighbour on both sides). D3 currently requires 6, so the first eligible keyframe moves from 6 to 7.
2. **Output syntax (D12):** LightEMMA asks for a list of tuples, `[(v1, c1), ...]`, not JSON. D12 said the format follows LightEMMA, so the templates use the list. The parser takes the last such list in the final answer.
3. **Speed timing (D12):** each speed applies over its whole 0.5 s step (history speeds are per-segment averages).
4. **Path conversion (D12):** LightEMMA uses a forward step (move along the old heading, then turn by curvature x speed x 0.5 s), which the plan rejected in favour of the exact arc.
5. **L2 (D13):** LightEMMA's "ADE 1 s / 2 s / 3 s" averages steps 1-2, 3-4 and 5-6 separately: a third convention, besides "average up to t" and "error at t".
6. **Ego pose source:** LightEMMA reads the ego pose attached to the CAM_FRONT sample data, not the LiDAR one. Using the same source keeps the history consistent with LightEMMA.
