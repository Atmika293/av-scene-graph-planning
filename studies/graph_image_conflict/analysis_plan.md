# Analysis plan: does the image correct a wrong scene graph?

**Status:** draft, not yet frozen. Freeze by committing this file with a dated tag (e.g. `plan-v1`) before any experimental run. Any decision added or changed after that tag is labelled exploratory in the write-up.

**Research question:** When a VLM planner's perception text leaves out a pedestrian or vehicle that its camera clearly shows, does it drive as if the object isn't there, or does what it sees rescue the plan?

**Setting and scope:** a zero-shot VLM planner (Gemma 4, D1) receives the nuScenes front camera image, its ego history and a scene graph built from the ground-truth annotations, in which exactly one object is wrong. The confirmatory tests (H1, H2; D14) cover **omissions**: the graph leaving out a movable object that is visible, near and in the car's path. **Additions** (phantoms) are a secondary question. Whether the plan moves the way the text points is reported through the direction measures (D13), not tested. (Revised 2026-11-27 from: "when a zero-shot VLM planner receives the nuScenes front camera image plus a scene graph, and the scene graph is wrong about one object, does the planned trajectory follow the text or the image?"; see `design_rationale.md`.)

Each decision records what was decided, when, and why. **The full reasoning** (the effects the design has to separate, the options considered for each decision, what changed during planning and the known limitations) is in `design_rationale.md`, frozen together with this plan; if the two disagree, this plan wins.

Decision numbers follow the Phase 0 list (D1-D16); sections are added as they are decided, so numbers may appear out of order until the plan is complete.

---

## Design overview

How the decisions fit together, in the order the pipeline runs:

1. **Frames:** all 150 val scenes (D2); keyframes 2 s apart with 3 s of history (7 keyframes before) and 3 s of future (D3).
2. **Clean scene graph** for each frame: objects in view with LiDAR points (D7, D8), fields and relations from the current and previous keyframe only (D8).
3. **Tags** on every object, for selection and grouping only, never shown to the model: visible (D7), near (D6), in-path now / entering within 3 s (D5).
4. **Edits (items)**, at most one of each kind per frame: critical drop, control drop, optional hidden drop (D9), phantom and optional control phantom (D10). Each broken graph differs from the clean one by exactly one edit.
5. **Conditions:** each frame and item is run under the conditions in D17, with repeats (D4), on each model (D1).
6. **Measuring and testing** (walkthrough section 6): plan changes measured from each condition's own base, floors from control drops (D4), metrics and averaging (D13), hypotheses (D14), tests (D15), invalid outputs (D12).
7. **Records** of every frame, object, item, call and run, so every selection can be reconstructed (D18).
8. **Checks before the full run:** automatic tests, visual review panels, a one-scene pilot on a train-split scene, then the plan is frozen (D19).

Future information (the driver's real path, objects' future positions) is used only to score plans and to label, select and group items; it never enters the model's input.

---

## Method walkthrough

A plain-language description of the whole process, step by step. The numbered decisions below are the formal specification: **if this walkthrough and a decision ever disagree, the decision wins.**

### 1. Choosing the frames (D2, D3)

The study uses all 150 scenes of the official nuScenes val split, each about 20 s long. nuScenes provides human-labelled 3D boxes only at keyframes, every 0.5 s. A keyframe is eligible if it has 7 keyframes (3.5 s) before it, for the model's 3 s of ego history (the extra keyframe is needed to compute the earliest turning value), and 3 s after it, to score the plan. Starting from the first eligible keyframe, one keyframe every 2 s is evaluated, usually 7 per scene, about 1,050 in total. Frames are never chosen by what they contain.

### 2. Building the clean scene graph (D7, D8)

For each frame, the scene graph is a short text description of the objects the front camera could show. It is built in five steps, using only the current keyframe (and the previous one, for speeds).

1. **Start from every labelled box** at the keyframe and map its category to one of the 10 nuScenes detection classes (car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, traffic cone, barrier). Other categories are dropped.
2. **Keep only objects in view of the front camera:** all 8 box corners in front of the camera, at least half of the projected box inside the image, and the visible part at least one visual token tall (about 36 px, measured exactly in Phase 2). The object must also have at least one LiDAR point. Partly occluded objects stay in: the graph imitates a camera-plus-LiDAR perception system limited to the front view. There is no cap on the number of objects.
3. **Describe each object:** class; position (metres ahead, metres left or right of the ego); distance (ego centre to the nearest point of the box); heading relative to the ego (same direction, opposite, crossing); speed from its movement since the previous keyframe; moving or stopped, from that speed. nuScenes' parked/stopped labels are not used, because annotators assigned them with the whole sequence in view.
4. **Relate each object to the ego vehicle only:** its bearing (ahead, ahead-left, ahead-right, left, right); whether it lies in a straight 2 m-wide-per-side corridor along the ego's current heading ("ahead in our direction"); and whether it is approaching or moving away.
5. **Write it as text:** one line per object, fields in a fixed order, nearest first, numbers rounded to 0.1, IDs assigned after sorting. An empty graph reads `No objects detected in view.`

An illustrative clean graph:

```
obj_1: pedestrian | 9.2 m ahead, 1.1 m right | 8.8 m | crossing | moving, 1.3 m/s | ahead in our direction | approaching
obj_2: car | 18.5 m ahead, 0.2 m left | 16.2 m | same direction | moving, 6.9 m/s | ahead in our direction | moving away
obj_3: car | 24.0 m ahead, 6.8 m left | 21.9 m | same direction | stopped, 0.0 m/s | ahead-left | approaching
```

### 3. Tagging the objects (D5, D6, D7)

Each object also gets tags that are **never shown to the model**. They decide which edits are allowed and how results are grouped.

- **Movable:** every class except traffic cone and barrier.
- **Visible:** in view, nuScenes visibility of 40% or more, at least one LiDAR point, so the image genuinely shows it.
- **Near:** within max(15 m, 3 s x the ego's current speed), the distance the ego could cover in the planning horizon.
- **In-path now / entering within 3 s:** the object overlaps a 2 m-per-side corridor around the driver's real path over the next 3 s, either now or at a later keyframe. This uses hindsight, which is acceptable because it only labels objects; the two tags are kept separate so their results can be compared.

### 4. Introducing the errors (D9, D10)

Each frame can produce up to three kinds of item (plus an optional fourth). Each item makes **exactly one** edit to a copy of the clean graph.

- **Critical drop:** among objects that are movable, visible, near and in-path, the **nearest** is deleted from the graph. This is the dangerous case: the text omits something in front of the car that the image clearly shows.
- **Control drop:** among movable, visible objects that are far or off-path, **one chosen at random** (with a recorded seed) is deleted. This error should barely matter; it checks whether the model reacts to any change in the text, relevant or not.
- **Optional hidden drop:** the nearest movable object that is near and in-path but mostly occluded (visibility under 40%) is deleted. The image cannot correct this one, so it separates genuine looking from a general distrust of text.
- **Phantom:** a stationary, median-sized car is added straight ahead along the ego's current heading, at a random distance between 10 m and min(25 m, the near distance). This is only attempted when the ego is moving faster than 2 m/s and its real path is nearly straight, so "straight ahead" reveals nothing about where the car will go. The phantom must be at least 1 m from every real box, pass the in-view rule, and be no more than 20% hidden in the image behind closer real objects, so the image clearly shows the spot is empty. If any check fails, the frame has no phantom; there are no redraws.

After each edit, the graph is re-sorted and IDs are reassigned, so no gap or position gives the edit away. A phantom's line is produced by the same code as real objects. An automatic check confirms the broken graph differs from the clean one only in that edit.

Example, using the graph above: the **critical drop** removes the crossing pedestrian (obj_1), so the text describes a clear road; the **control drop** might remove the stopped car to the left (obj_3); a **phantom** might add a stationary car 14 m straight ahead, which the image shows to be empty road.

### 5. Asking the models (D1, D4, D17)

Every request contains, in this order: the front camera image (or a uniform gray image, or no image), the ego vehicle's last 3 s of speed and turning, and the scene graph (correct, broken, or none), followed by the planning instruction. Each model reasons step by step, then gives its final answer as 6 pairs of speed and curvature, one per 0.5 s for the next 3 s, which fixed code converts into a planned path of 6 positions (D12).

**What the answer looks like.** After its reasoning, the model must end with one list of six (speed, curvature) pairs, in LightEMMA's format:

```
[(8.1, 0.000), (8.3, 0.002), (8.4, 0.004), (8.4, 0.004), (8.5, 0.003), (8.5, 0.002)]
```

Each pair is (speed in m/s, curvature in 1/m) for one 0.5 s step. Curvature is 1 divided by the turning radius: 0 is straight, 0.02 a gentle curve (50 m radius), 0.1 a tight turn (10 m radius). Positive curvature turns left, negative turns right.

**From pairs to a path.** Starting at the car's current position and heading, each step covers speed x 0.5 s along a circular arc of the given curvature, turning the heading by curvature x distance. The result is 6 positions (x ahead, y left), each measured from where the car is **now**, in a frame frozen at this moment, not from the previous step. The driver's real future positions are converted into the same frame, so planned and real paths can be compared directly.

**Invalid answers.** An answer that cannot be parsed, is cut off, does not have exactly 6 pairs, or asks for impossible values (speed below 0 or above 30 m/s, curvature beyond 0.2 per metre) is invalid. It is counted, never retried (D12).

- **Clean conditions run on every frame:** image only and ego history only (once each); image + correct graph, gray + correct graph and graph only (3 times each).
- **Corrupted conditions run on every item:** image + broken graph, gray + broken graph and graph only with the broken graph (3 times each).
- Both Gemma models run everything, with Google's recommended sampling and a recorded seed for every request. The three repeats are combined by their median path (the mean is also reported).

### 6. Measuring and testing (D4, D12, D13, D14, D15)

#### 6.1 From answers to paths

Each condition is run 3 times. Each run's answer is converted into 6 positions (section 5). The three runs are combined into one path by taking, at each of the 6 times, the median of the three "ahead" values and the median of the three "left" values. Medians are taken over positions, never over the raw speeds and curvatures, because the median of those, once converted, can be a path that no run actually planned. The mean path, computed the same way, is used as a robustness check. A condition needs at least 2 valid runs; otherwise it is invalid (D12).

#### 6.2 How far an error moves the plan

**Plan change** between two paths is the average distance between their positions at the 6 times, in metres. It counts any change: steering differently, driving faster or slower, or both.

For every item, three plan changes are computed. Each compares the broken-graph path with the correct-graph path **under the same input**, so the only difference between the two paths is the edit:

| Plan change | Broken path | Measured from (base) | Question |
| --- | --- | --- | --- |
| Text-only | graph only, broken | graph only, correct | Does the error matter when the text is all the model has? |
| Real-image | image + broken | image + correct | How far does the error pull the full planner off its own correct plan? |
| Gray-image | gray + broken | gray + correct | The same, with an image slot that shows nothing |

Measuring each from its own base matters. Adding an image changes the plan even when the graph is correct (the image may show the road curving, for example), so measuring a gray-image path from the real-image base would count that difference as if the error had caused it.

Each plan change is split into two parts that show what kind of change happened, plus a safety view:

- **Heading divergence (direction):** at each 0.5 s step, the angle between the two paths' directions of travel. Both paths start at the car's current heading, so they share a base. Steps where the car moves less than 1 m (under 2 m/s) are skipped, because a direction means little when the car barely moves. Summarised as the **largest** absolute angle over the steps, in degrees (the peak swerve), with the average also recorded and the left/right sign kept separately. Looking at every step catches a swerve that straightens out, which the heading at 3 s alone would miss. It is reported for critical and control drops side by side, so the control value shows what "no real effect" looks like; it has no floor, because nothing is decided by it.
- **Change in progress (speed):** how much further (positive) or less far (negative) the broken path travels in 3 s.
- **Change in clearance (safety):** how much closer (negative) or further (positive) the broken path comes to the altered object.

Heading divergence alone would miss the most common reaction: when a pedestrian ahead is dropped, the plan usually speeds up rather than steers, so the heading divergence is near 0 while progress differs by metres. That is why the tests use plan change, which covers both.

**Expected direction.** For a drop, the dangerous direction is more progress and less clearance: driving on towards the object the text left out. For a phantom, the expected direction is less progress and more clearance: braking for a car that is not there. Among items misled with the real image, each component (progress, clearance) is labelled **expected**, **opposite** or **no clear change**; a change counts as clear only if it is larger than the 95th percentile of that component's size over control drops, so a tiny change is not counted by its sign alone. Reported as two shares (progress, clearance) plus the share where both are clear and expected; the sign-only share (50% = no direction) is reported alongside as a robustness check. A high expected share shows the plan change reflects the model acting on the false text, not random drift.

#### 6.3 What "no effect" looks like: floors from control drops

A plan change is never exactly zero. A critical drop's plan change contains three things:

> plan change = run-to-run noise + effect of editing the text + effect of the error itself

Two medians of 3 runs always differ a little (noise), and editing the text at all can nudge the plan (a line is removed and the IDs are renumbered).

The **control drop** works as a placebo: the same procedure as a critical drop, applied to an object that should not matter (far away or off the path). Its plan change contains the noise and the edit effect, and close to no error effect. It is also measured exactly like a critical item: median of 3 against median of 3, under the same conditions.

For each model and each of the three plan changes, the **floor** is the 95th percentile of that plan change over all control drops. Floors are computed by a fixed automatic step, **before any H1 or H2 value is looked at**, and recorded.

The floors are used only as cutoffs for single items (6.4). The tests use the control drops' averages instead (6.6).

Why not the baseline's own repeats: with 3 runs per condition there is one median per frame, so the repeats only show how far single runs differ from each other, which is more than two medians differ. A baseline-only floor at the right level would need 6 runs per condition, and it would still miss the edit effect. The spread between single runs is reported anyway, as a description of how much the model varies.

Example (illustrative numbers): control drops' real-image plan changes mostly fall between 0.2 and 0.6 m, with a 95th percentile of 0.7 m. A critical drop at 1.2 m is beyond what a no-effect edit produces; one at 0.5 m is not.

#### 6.4 Eligibility and the misled rate

**Eligible:** a critical drop whose text-only plan change is above the text-only floor. The error changes the plan when the model has only the text, so once the image is added there is a genuine conflict between text and image. Ineligible items are left out of H1, H2 and the misled rate, because they would add near-zero values that say nothing about the image. Their share is reported as a finding: a large share means the model largely ignores the graph.

Eligibility is decided on the graph-only runs, which are separate from the image and gray runs that the tests measure, so selecting items cannot build an effect into the result.

**Misled rate:** among eligible items, the share whose real-image plan change is above the real-image floor: the error pulled the plan off course even though the image could show the truth. The same share with the gray image gives the comparison. It is the planning counterpart of the measure in Deng et al., "Words or Vision" (how often the answer follows the text when text and image conflict), and it **leads the write-up**, e.g. (illustrative numbers): "with the real image, 38% of errors still pulled the plan off course, against 61% with the gray image; by construction about 5% of control edits cross the floor."

It is not used for the tests: turning each item into yes or no misses partial correction (an error that moves the plan 3 m with the gray image and 1 m with the real image counts as misled both times), depends on where the floor sits, and makes the test less sensitive.

#### 6.5 Averaging per scene

Item values are averaged within each scene, then across scenes, so every scene counts once whatever its number of items; frames from the same scene are not independent. Frame-pooled averages are also reported, for comparison with other planning papers.

#### 6.6 H1 and H2

Both use **critical drops** on **Gemma 4 31B**, eligible items only. Phantoms are reported separately (6.7).

**H1: does the error get through when the real image is present?** For each scene:

> (average real-image plan change on eligible critical drops) minus (average real-image plan change on control drops)

The critical average contains noise + edit effect + error effect; the control average contains noise + edit effect + close to nothing. The difference is the error's effect. Example: critical 1.4 m, control 0.4 m, so the scene's value is 1.0 m.

**H2: does the image's content correct it?** For each scene:

> (average of gray-image minus real-image plan change on eligible critical drops) minus (the same average on control drops)

The first part is how much less the error moves the plan when the image shows the scene. The control part removes a general effect: without real content to look at, the model may lean on the text more, so even irrelevant edits may move gray-image plans more than real-image ones. What remains is the correction specific to the relevant error: the object in the image contradicts the text. The first part alone (unadjusted) is also reported. If it is clearly positive while the adjusted value is near zero, the image lowers trust in the text in general without correcting specific errors.

**The test.** Each hypothesis's per-scene values are tested against zero with a two-sided sign-flip permutation test: if the true effect were zero, each scene's value would be equally likely to be positive or negative, so flipping the signs at random 100,000 times (fixed seed) shows how large an average arises by chance. Scenes without the items needed are excluded and counted. Effect size: the average per-scene value, with a 95% interval from resampling scenes 10,000 times.

**Two hypotheses, one 5% false-positive rate (Holm).** Both are tested. The smaller p-value must be below 0.025 and the other below 0.05. Unlike a fixed sequence (H2 only if H1 passes), this keeps H2 testable when H1 fails because the image fully corrects the error, which is exactly when H2 matters most.

**Reading the outcomes** (written before any data):

| H1 | H2 | Reading |
| --- | --- | --- |
| supported | supported | Errors get through, but the image's content reduces them. |
| supported | not supported | The model follows the text even when the image contradicts it. |
| not supported | supported (items eligible) | The image's content fully corrects the error. |
| not supported | not supported (items eligible) | Any image in the prompt, even an empty one, keeps the error from moving the plan (compare S4). |
| not supported | n/a (most items ineligible) | The model largely ignores the scene graph (see S3). |

**Robustness (no tests):** the same values with the mean path instead of the median, with per-scene medians instead of averages, and H2 without the control adjustment.

#### 6.7 Secondary results and questions

Reported with 95% intervals and no tests. Each breakdown gets the real- and gray-image plan changes, the misled rate, heading divergence, changes in progress and clearance, and the expected-direction share.

| Question | What answers it |
| --- | --- |
| Phantoms | The breakdown measures above, never pooled with drops (their expected direction is the opposite). Floors come from control drops, a stated caveat unless the optional control phantom (D10) is run. |
| Gemma 4 E4B | The breakdown measures above; H1 and H2 values with intervals |
| In-path now vs entering within 3 s | The breakdown measures above, per tag |
| Vehicles vs vulnerable road users | The breakdown measures above, per group (exploratory) |
| Hidden drop (if run) | The image cannot show the object, so real- and gray-image plan changes should be about equal. A remaining difference is the image's general effect on trust in the text: an independent check on H2's control adjustment. |
| Position in the graph text | Control-drop plan change for objects early vs late in the list. A critical drop usually removes the first line and renumbers every ID; if position matters, the edit effect differs between critical and control drops. |
| Closed model (if run) | The breakdown measures above |
| **S1:** does any input beat simple extrapolation? | L2 and collision: clean model conditions vs constant velocity and constant curvature |
| **S2:** does the correct graph add anything beyond the image? | L2 and collision, image only vs image + correct; plan change between them |
| **S3:** how much of planning is continuing past motion? | L2 and collision, ego history only vs the others; plan change of each condition from ego history only |
| **S4:** does an empty image slot change behaviour? | Plan change, gray + correct vs graph only, correct (the floors give the scale); L2 |

#### 6.8 Every metric and its role

Only plan change enters the hypothesis tests. A few other things decide **which items enter** them: the eligibility gate (built on the text-only plan change and its floor) and the invalid-output rule (D12). Everything else explains the results: whether the plan steered or braked, came closer or stayed away, how often errors got through, and how well the plans drive. None of it can change whether H1 or H2 is supported.

Not every metric exists for every condition. Metrics on **one path** are computed for every condition, including image only, ego history only and the two simple baselines. Metrics that **compare two paths** (the broken one and the correct one under the same input) exist only for items, three times each: text only, real image and gray image.

**Metrics on one path**

| Metric | Computed for | Role |
| --- | --- | --- |
| L2 vs the driver's path (two conventions, plus the LightEMMA comparison column) | Every condition, including image only, ego history only, constant velocity and constant curvature | Reporting: S1 to S3; broken conditions shown next to their correct-graph versions |
| Collision with real objects | Every condition | Reporting: as for L2 |
| Progress (path length in 3 s) | Every condition | Reporting; feeds the change in progress |
| Clearance to the altered object | The correct and broken paths of each item | Reporting; feeds the change in clearance |
| Invalid-output rate | Every model condition | Reporting, **plus an inclusion rule:** items needing an invalid condition are left out of the plan-change measures (D12) |
| Spread between single runs | Conditions with 3 repeats | Reporting: how much the model varies from run to run |

**Metrics comparing two paths (broken vs correct, same input)**

| Metric | Role |
| --- | --- |
| Plan change, real image | **Test:** H1, H2 |
| Plan change, gray image | **Test:** H2 |
| Plan change, text only | **Gate:** decides eligibility; its size is reported |
| Floors (95th percentile of control drops' plan changes, D4) | **Threshold:** eligibility and the misled rate |
| Misled rate (real image, gray image) | Reporting: the headline result, leads the write-up |
| Heading divergence | Reporting: did the plan steer differently? |
| Change in progress | Reporting: did the plan speed up or brake? |
| Change in clearance | Reporting: did the plan come closer to the object? |
| Expected-direction share | Reporting: do the changes go the way the error points? |

Plan change is also computed between clean conditions for S2 to S4 (for example image only vs image + correct graph); that is reporting too.

**Other**

| Metric | Role |
| --- | --- |
| Reasoning mentions the altered object | Exploratory: a hand-checked sample of items |

### 7. Keeping a record (D18)

Every frame, every annotated object (including why it was left out of the graph), every edit made or impossible, every model call and every run is saved, so each selection and each number can be traced back to the data.

---

## D1. Models

**Decided:** 2026-11-24, revised 2026-11-25 (switched from Claude models to open-weight Gemma 4 models, before any run)

| Role | Model | Weights | Exact revision (Hugging Face commit, recorded at first run) |
| --- | --- | --- | --- |
| Primary | Gemma 4 31B (dense) | `google/gemma-4-31B-it`, official, BF16 | _to fill in_ |
| Secondary | Gemma 4 E4B (edge-sized) | `google/gemma-4-E4B-it`, official, BF16 | _to fill in_ |
| Optional (exploratory, only if time allows after the main runs) | Claude Sonnet 5.5 | closed, `claude-sonnet-5-5` | _version string if run_ |

- **The primary hypothesis is tested on the primary model only.** The secondary model runs the same frames, conditions and prompts and is reported as a comparison of model size within one family.
- **The optional closed model** is listed now so that adding it later is not a choice made after seeing results. If run, it uses the same prompts and the closest available reasoning setting, and is reported as exploratory.

**Serving (identical for both Gemma models):**

- Official weights at full precision (BF16), served with vLLM on rented RunPod GPUs. No quantised variants and no third-party pay-per-token providers, whose precision cannot be verified.
- Recorded for every run: vLLM version, Docker image tag, Hugging Face model revision (commit hash), GPU type.

**Generation settings (identical for both Gemma models):**

- **Sampling:** Google's recommended standard configuration for Gemma 4: temperature 1.0, top-p 0.95, top-k 64.
- **Seeds:** every request carries its own seed, derived deterministically from its request ID, and the seed is logged.
- **Reasoning on** for both models (Gemma 4's built-in thinking mode), because a planner is expected to reason before acting. The reasoning text is saved for every call; the planned path is parsed from the final answer only.
- **Image budget:** 1120 visual tokens per image (Gemma 4's highest budget), for both models, so small and distant objects keep as much detail as possible.
- **Input order:** image first, then ego history, then scene graph (Google's recommended order for Gemma 4, and fixed for every condition).
- **Maximum output length:** fixed from the one-scene pilot before any full run, set generously (expected about 4,096 tokens). An answer cut off at the limit counts as an invalid output.

**Why these models:**

- **Open weights make the study verifiable:** the architecture, parameter counts and exact weights are public, so a reader knows exactly what was tested, and anyone can rerun it permanently.
- **They allow later inspection of the model's internals** (for example, how much attention goes to image tokens versus scene-graph tokens under conflict), which a closed model does not.
- **Same family, different size:** 31B dense versus E4B ("effective 4B", designed for phones and edge devices) isolates model size while keeping training approach and prompt format the same, and asks whether a model small enough to run on a vehicle behaves differently. The vision encoders also differ (about 550M versus about 150M parameters), so this is a comparison of whole models, which the write-up states.

**Scope:** results apply to Gemma 4 31B and E4B. Closed frontier models are untested unless the optional run is done.

**Cost:** GPU time on RunPod, measured in the one-scene pilot. **Budget cap:** $100 (D16).

---

## D2. Scenes

**Decided:** 2026-11-24

- **Source:** the official nuScenes val split, all 150 scenes, taken from the ingested trainval tables using the devkit's split list (`nuscenes.utils.splits.create_splits_scenes()["val"]`), matched on scene name.
- **Why all 150:** results are resampled at the scene level, so the number of scenes sets precision; all 150 also gives the widest variety of situations, needs no sampling choice to defend, and matches LightEMMA's evaluation set.
- **Why val:** the test split has no public 3D annotations, which the scene graph, visibility tags and collision check need. Train stays untouched for the later detector study (Build 2), so the planner is never tested on scenes a detector learned from.
- **Saved list:** the 150 scene names are written to `scenes.txt` in this folder and committed with this plan.
- **Budget fallback (rule, not judgement):** if the one-scene pilot projects cost above the budget cap, scenes are dropped in a random order fixed by seed 0, never chosen by hand.
- **Unusable scenes:** a scene with no eligible frames under D3 is excluded and reported, with the reason.

---

## D3. Frames

**Decided:** 2026-11-24, revised 2026-11-27 (history length from the LightEMMA check)

- **Keyframes only** (nuScenes samples, 0.5 s apart): 3D boxes exist only at keyframes.
- **History and future come from all keyframes at 0.5 s.** The spacing below decides only which keyframes are evaluated, not where history or future is read from.
- **Eligible:** a keyframe with 7 keyframes before it and 6 after it (3 s of future). The history is 6 speeds and 6 curvatures at 0.5 s (3 s); the extra keyframe is needed because each curvature uses a pose on either side (LightEMMA's format). In a scene with *n* keyframes, that is keyframes 7 to *n* - 7: the first 3.5 s and last 3 s of each scene are never evaluated.
- **Selected:** every 4th eligible keyframe (one every 2 s), starting from the first eligible one (keyframe 7), **with no cap per scene**: every keyframe on that 2 s grid that meets the history and future rule is used. A typical 40-keyframe scene gives 7 frames (keyframes 7, 11, 15, 19, 23, 27, 31); longer scenes give 8, shorter ones 6.
- **Blind to content:** frames are never chosen or dropped by what they contain. Which frames qualify for corruption is a separate rule, applied to this fixed set.
- **Paired:** the same frames are used for every condition and both models.
- **Expected size:** about 7 per scene on average, about 1,050 frames in total; the exact count, and the number of frames per scene, come from Phase 1 and are reported.
- **Why 2 s spacing:** keyframes 0.5 s apart are near-duplicates and add cost without independent information; the scene is the unit of analysis, so variety across scenes matters more than frames per scene. Rejected alternatives: every 2nd keyframe (about 2,100 frames, about double the cost) and every 6th (fully independent future windows, but only 4-5 frames per scene).
- **History values (LightEMMA's format, checked 2026-11-27 against its code, commit 4906a89):** 8 ego poses (from 7 keyframes before the current one up to the current one), taken from the ego pose attached to the CAM_FRONT sample data and transformed into the current ego frame; 6 speeds (distance between consecutive poses divided by their time difference; values below 0.2 m/s set to 0) and 6 curvatures (circle through each pose and its two neighbours, signed, positive = left; values beyond +/-0.2 per metre set to 0), rounded to 3 decimals and written as lists, most recent last. Details in `prompts/README.md`.

---

## D4. Repeats and noise floor

**Decided:** 2026-11-25 (repeats), 2026-11-26 (combining repeats as positions; floors from control drops)

- **3 repeats** of every condition that is compared frame by frame: image + correct graph, gray image + correct graph, correct graph only, and their three broken-graph counterparts. Each repeat has its own seed.
- **1 run** of image only and ego history only: these are only averaged across frames, never compared frame by frame.
- **Combining repeats:** each repeat is first converted to 6 positions (D12). The **median** path is then taken separately for each coordinate (ahead, left) at each of the 6 time steps, across the 3 repeats; this is primary. The **mean** path, taken the same way, is also recorded and reported as a robustness check. Medians are never taken over the raw speed and curvature values, because the median of those, once converted, can be a path that no repeat actually planned.
- **Distance between two paths:** the mean Euclidean distance between their positions at the 6 time steps (= plan change, D13).
- **Floors (from control drops):** for each model and each of the three plan changes (text-only, real-image, gray-image; D13), the floor is the **95th percentile of that plan change over all control drops**. Floors are computed by a fixed automatic step **before any H1 or H2 value is computed**, and are recorded (D18).
- **Use of the floors:** only as cutoffs for single items: eligibility (text-only floor) and the misled rate (real-image and gray-image floors; D13). The tests do not use the floors; they use the control drops' averages (D15).
- **Spread between single runs:** the 95th percentile of distances between pairs of single repeats of the same condition on the same frame, reported per model and per condition as a description of the model's variability. Not used as a cutoff.

**Why 3 repeats:** with sampling at temperature 1.0, outputs vary between identical requests; 3 repeats give a better estimate of that variation and a central path that a single odd output cannot distort.

**Why the floor comes from control drops:** a critical drop's plan change compares a median of 3 with a median of 3, and contains run-to-run noise, the effect of editing the text at all (a line removed, IDs renumbered) and the error's own effect. A control drop goes through the same procedure with an object that should not matter, so it measures the first two at exactly the same level, with no extra model calls. Distances between single repeats are larger than distances between medians, so a floor built from them would be too high, and they miss the edit effect; a baseline-only floor at the right level would need 6 repeats per condition. If a control object matters slightly, the floor comes out slightly high, which errs on the cautious side.

---

## D5. In-path

**Decided:** 2026-11-25

**Used for:** deciding which edits are allowed (a critical drop needs a near, in-path, visible object; a control drop needs a far or off-path visible object) and grouping results (the primary hypothesis is tested on critical items; control items check whether the model reacts to any change in the text, relevant or not). **The label is never shown to the model.**

- **Path:** the driver's real path over the next 3 s: the 6 future ego poses, transformed into the ego frame at the current keyframe.
- **Corridor:** a polyline from the ego's current position through those 6 points, **2 m wide on each side** (about half the vehicle's width plus a margin, roughly one lane), extended straight along the final heading out to at least the near distance from D6, so a slow or stopped vehicle still has a corridor ahead.
- **Test:** an object is in-path if its bird's-eye-view box footprint overlaps the corridor **now or at any of the next 6 keyframes** (using that object's own future annotations; keyframes where the object is not annotated are skipped). This includes objects about to enter the path, such as a pedestrian stepping off the kerb.
- **Two tags, recorded separately:** *in-path now* (overlaps the corridor at the current keyframe) and *entering within 3 s* (does not overlap it now, but does at a later keyframe). Either tag makes an object eligible as in-path for item selection.
- **Why two tags:** *entering within 3 s* uses hindsight. Some of these objects are foreseeable from the current frame (a pedestrian at the kerb walking toward the road); others are not (a parked car that pulls out a moment later). Future information is used only to label, select and group items, never to build the model's input, so nothing leaks; but the two groups may behave differently. Results for critical items are therefore also reported broken down by tag.
- **Limitation:** the corridor follows what the driver actually did, so an object the driver steered around (e.g. a parked car passed by changing lane) may fall outside it. It also ignores timing: an object that enters the corridor after the ego has passed still counts. A map-based lane definition is a next step.
- **Check:** for 10 frames, draw the corridor and all boxes on the bird's-eye-view render and confirm by eye which objects are tagged in-path.

---

## D6. Near

**Decided:** 2026-11-25

- **Near = within max(15 m, 3 s x current ego speed)**, measured from the ego vehicle's centre to the nearest point of the object's bird's-eye-view box footprint at the current keyframe.
- **Current ego speed** comes from the current and previous ego poses only (no future information).
- Examples: stopped or below 5 m/s, 15 m; 10 m/s, 30 m; 15 m/s, 45 m.

**Why horizon-based:** the model plans 3 s ahead, so anything the ego could reach within that horizon can affect the plan; a fixed distance would be too generous in slow traffic and too short at speed. The 15 m minimum keeps nearby objects relevant when the vehicle is stopped or slow.

**Used for:** together with D5 and D7, defining critical items (near, in-path and visible) and control items (visible, and far or off-path).

---

## D7. In view and visible

**Decided:** 2026-11-25, revised 2026-11-25 (height threshold changed from Claude's 28 px patch to Gemma 4's visual-token footprint)

Two layers, defined once here and reused by the scene-graph, drop and phantom rules.

**In view (could the front camera show it?)** An object is in view if all of these hold:

- all 8 corners of its 3D box are in front of the front camera (positive depth);
- at least 50% of its projected 2D box (the box around the 8 projected corners) lies inside the 1600 x 900 image;
- the part of that projected box inside the image is **at least as tall as one visual token's footprint** at the image budget in D1 (1120 tokens). For an unresized 1600 x 900 image this is about sqrt(1600 x 900 / 1120), roughly **36 px**; the exact value is measured from Gemma 4's image processor in Phase 2 and recorded here before freezing: _measured value: to fill in_.

Projection uses the validated Phase 2 camera code and the real calibration, so the view starts at the camera, not the vehicle's centre. **There is no distance cutoff:** whether the model can see an object depends on its size in the image, not its distance. Objects not in view never appear in the scene graph.

**Visible (is it actually seen, or blocked?)** An object is visible if it is in view, has a nuScenes visibility of 40% or more (bins 40-60%, 60-80%, 80-100%), and has at least one LiDAR point.

**Why these thresholds:**

- The height threshold is tied to how the primary model sees the image: an object shorter than one visual token's footprint cannot be resolved as a separate object. Both Gemma models use the same image budget, so the threshold is the same for both. It is also in the range of the 25 px minimum box height used by the KITTI benchmark for its moderate and hard levels (to verify before citing).
- Height rather than area, so thin objects such as pedestrians are not penalised.
- 50% inside the image excludes objects mostly cut off at the image edge.
- The scene graph must never describe anything the front image cannot show; otherwise graph-only conditions would have an information advantage over the image.
- Visibility separates objects the image could correct (visible) from those it could not (in view but occluded), which is the hidden-drop control.

**Check:** for 10 frames, draw the projected boxes on the front image and the camera's view cone (angle computed from the calibration) on the bird's-eye-view render, and confirm by eye which objects pass each rule.

---

## D8. Scene-graph contents

**Decided:** 2026-11-25

**Objects considered:** every human-labelled 3D box at the current keyframe, mapped to the 10 classes of the nuScenes detection benchmark: car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle, traffic cone, barrier. Categories outside these (animals, debris, bicycle racks and similar) are left out.

**Objects included:** those **in view** (D7) with **at least one LiDAR point**. They need not be visible, so partly occluded objects are included. The graph imitates a camera-plus-LiDAR perception system limited to the front camera's view; including occluded objects also makes the hidden-drop control possible.

**No cap on object count:** every qualifying object is included, however many there are; the number of objects per graph is recorded. A cap would make the graph omit objects the image shows, a mismatch unrelated to the deliberate errors. If the pilot shows very long graphs, any cap must be set by a written rule before the full run.

**Fields per object, from the current keyframe (plus the previous keyframe for speed only):**

| Field | How it is computed |
| --- | --- |
| Class | the 10-class mapping above |
| Position | box centre in the ego frame: metres ahead, metres left or right |
| Distance | ego centre to the nearest point of the box footprint (same as D6) |
| Heading | box direction relative to the ego: same direction, opposite, or crossing |
| Speed | displacement between the previous and current annotation over 0.5 s; "unknown" if the object has no previous annotation |
| Moving or stopped | from that speed (moving above about 0.5 m/s) |

nuScenes' labelled attributes (such as parked or stopped) are **not** used: annotators assigned them with the whole sequence in view, so they can carry hindsight.

**Relationships: ego-centric only** (each object relates to the ego vehicle, never to other objects):

- Bearing: ahead, ahead-left, ahead-right, left or right.
- *Ahead in our direction:* the box overlaps a straight corridor 2 m either side along the ego's **current** heading. This is deliberately not the D5 in-path label, which uses the driver's future path.
- Approaching or moving away: distance shrank or grew since the previous keyframe.

**Text format:** one line per object, fields always in the same order, sorted nearest first, numbers rounded to 0.1, IDs (`obj_1`, `obj_2`, ...) assigned after sorting. If no object qualifies, the graph reads `No objects detected in view.`

**No future information:** nothing in the graph uses any keyframe after the current one.

---

## D9. Drop rules

**Decided:** 2026-11-25

Object tags used here (never shown to the model): movable (every class except traffic cone and barrier), visible (D7), near (D6), in-path now / entering within 3 s (D5).

- **Exactly one edit per item.** Each frame can have at most one item of each kind.
- **Critical drop:** candidates are movable, visible, near and in-path (either tag). The **nearest** candidate is dropped. No candidate, no critical drop for that frame.
- **Control drop:** candidates are movable, visible, and not near or not in-path. One is chosen **at random**, with the random draw seeded from seed 0, the frame's sample token and the item kind. No candidate, no control drop.
- **Optional hidden drop (exploratory, only if time allows):** candidates are movable, in view, with at least one LiDAR point, visibility under 40%, near and in-path. The nearest is dropped. This is the case the image cannot correct.
- **Editing:** the object's line is removed (with ego-centric relations, that removes everything about it), the graph is re-sorted and IDs are reassigned, so no gap or position gives the edit away.
- **Recorded per item:** frame, kind, object ID and class, distance, all tags, random draw.

---

## D10. Phantom rule

**Decided:** 2026-11-25

- **Only on frames where** the ego is faster than 2 m/s and the driver's path turns by less than 10 degrees over 3 s (so straight ahead and the driver's path coincide, and placing the phantom straight ahead leaks nothing about the future path).
- **Object:** one stationary car, sized at the median of all nuScenes car boxes in the val split (computed once and recorded), facing the ego's current heading.
- **Placement:** straight ahead along the ego's current heading, at a distance drawn at random between 10 m and min(25 m, the D6 near distance), so every phantom is near and in-path. The draw is seeded from seed 0, the frame's sample token and the item kind.
- **It must be checkably wrong; all checks must pass:**
    - its footprint is at least 1 m from every real box (bird's-eye view);
    - it passes the D7 in-view rule;
    - **not hidden in the image:** its projected 2D box is covered by no more than 20% by the combined projected 2D boxes of real objects that are closer to the camera than it is, so the image clearly shows the spot is empty.
- **If any check fails,** no phantom for that frame, and the reason is logged. No redraws.
- **Editing:** the phantom's line is created by the same code as real objects, inserted, and the graph re-sorted with IDs reassigned.
- **Recorded per item:** frame, drawn distance, phantom position and size, the result of each check.

- **Optional control phantom (exploratory, only if time allows):** on the same frames as the phantom, the same car at the same drawn distance ahead, but shifted sideways outside the path corridor, its centre 5 to 8 m to a randomly chosen side (draw seeded from seed 0, the frame's sample token and the item kind). It must pass the same checks (at least 1 m from every real box, in view, at most 20% hidden by closer objects); if any fails, no control phantom for that frame, no redraws. It is an irrelevant false addition: the placebo for phantoms, as the control drop is for drops.

**Checks for D9 and D10:** an automatic test confirms each broken graph differs from the clean one only in the edited line and the renumbering; by eye, for 10 items, all real boxes and the phantom are drawn on the front image.

---

## D12. Output format and invalid outputs

**Decided:** 2026-11-26, revised 2026-11-27 (output format, speed timing and conversion settled by the LightEMMA check)

**Expected output.** After its reasoning, the model's final answer must be one list of 6 (speed, curvature) pairs, in LightEMMA's format:

```
[(8.1, 0.000), (8.3, 0.002), (8.4, 0.004), (8.4, 0.004), (8.5, 0.003), (8.5, 0.002)]
```

- Six pairs of (speed in m/s, curvature in 1/m), one for each 0.5 s step over the next 3 s; positive curvature turns left.
- This matches the format of the ego history the model receives (speed and curvature) and LightEMMA's output format (checked 2026-11-27 against LightEMMA's code, commit 4906a89). The prompt templates are in `prompts/`.
- **Conversion to a path:** fixed, tested code integrates the pairs into 6 positions, in 0.5 s steps. For each step: distance = speed x 0.5 s; heading change = curvature x distance; the ego then moves along an **exact circular arc** of that curvature (no approximation, since speed and curvature are constant within a step). LightEMMA converts with a **forward step** (move straight along the old heading, then turn); that is used only for the LightEMMA comparison column in D13, never for our own measures, because on a gentle left curve at 8 m/s it ends almost 1 m off sideways after 3 s.
- **Speed timing:** each speed applies across its whole 0.5 s step (LightEMMA's history speeds are averages over each step).
- **Frame:** every position is measured from the ego's position and heading at the current keyframe, in a frame **frozen at that moment** (x ahead, y left). Positions are cumulative from that origin, not relative to the previous step. The ground-truth future poses are converted from nuScenes world coordinates into the same frame. All metrics use these positions.
- **Parsing:** the reasoning block is set aside (and saved); the **last** list of pairs in the final answer is used.
- **No constrained decoding:** the model is not forced into the list format, so the invalid-output rate is a genuine measurement.

**Invalid repeat.** A repeat is invalid if any of these hold:

- no final answer could be parsed (no list of pairs, wrong shape, or the answer was cut off at the output limit);
- not exactly 6 pairs;
- any value missing or not a number;
- physically impossible: speed below 0 or above 30 m/s, or curvature above 0.2 per metre in magnitude (a turning radius under 5 m).

**What happens next:**

- **No retries:** invalid answers are counted, not replaced by asking again.
- **A condition needs at least 2 of its 3 repeats valid;** its central path is the median (and mean, D4) of the valid repeats. Conditions run once (image only, ego history only) are invalid if that run is invalid.
- **L2 and collision (primary):** an invalid condition is scored as the constant-velocity path, so failures stay in the average at the level of a trivial fallback instead of disappearing (dropping them would score the model only on the frames it managed, often the easier ones). **Valid-only results are reported alongside.**
- **Plan changes and everything built on them:** an item is excluded from any comparison that needs an invalid condition, and counted (a fallback path would create a fake plan change). The same applies to control drops in the floors and in the control averages of H1 and H2.
- **Invalid rates** are reported per model and condition.

---

## D13. Metrics

**Decided:** 2026-11-25 (averaging), 2026-11-26 (metric definitions; aggregation over the 6 positions; the text-following ratio replaced by plan changes measured from each condition's own base)

### Order of aggregation

1. **Repeats to one path:** each repeat is converted to 6 positions in the frozen current-ego frame (D12); the median across repeats is taken per coordinate and per time step (D4).
2. **6 positions to one number per item:** L2 is not combined (reported at, and averaged up to, 1, 2 and 3 s); collision = any step; plan change = mean over the 6 steps; heading divergence = largest absolute angle over the usable steps (mean also recorded); clearance = smallest over the 6 steps; progress = total path length.
3. **Items to scenes:** mean within each scene.
4. **Scenes to the result:** mean across scenes (primary); frame-pooled mean (secondary). See below.

### Metric definitions

| Metric | Definition |
| --- | --- |
| L2 | Distance between planned and real ego position at each 0.5 s step up to 3.0 s, always reported under **both** conventions: the average over steps up to t, and the error at t. L2 is not part of the hypothesis tests, so neither convention is primary. **For comparison with LightEMMA only,** L2 is also computed their way: their forward-step conversion and their averages over steps 1-2, 3-4 and 5-6 ("ADE 1 s / 2 s / 3 s"); this column is used for nothing else. |
| Collision | The ego footprint (4.08 x 1.85 m, the nuScenes vehicle), placed at each planned position and facing along the path, is checked for overlap with every real box at the same future keyframe (bird's-eye view, polygon overlap). A frame collides if any step overlaps. Frames where the driver's own path overlaps a box are excluded from this metric. Reported as counts with exact (Clopper-Pearson) intervals. |
| Plan change | Mean distance between the positions of two paths at the 6 time steps, in metres. Unsigned: any change counts, whatever its direction. |
| Text-only, real-image and gray-image plan change | For each item, the plan change between the broken-graph and the correct-graph path **under the same input**: graph only, broken vs graph only, correct; image + broken vs image + correct; gray + broken vs gray + correct. Each is measured from its own correct-graph base, so the two paths differ only in the edit. |
| Heading divergence | At each 0.5 s step, the angle between the two paths' directions of travel (from one position to the next; from the current position for the first step). Both paths start at the current heading. Steps where either path moves less than 1 m are skipped. Summarised as the **largest** absolute angle over the remaining steps, in degrees; the mean absolute angle and the signed mean (positive = the broken path heads more to the left) are also recorded. Reported for critical and control drops side by side; no floor, since nothing is decided by it. The direction part of a plan change. |
| Progress | Distance travelled along the planned path by 3 s (braking shows up here). |
| Change in progress | Progress of the broken path minus progress of the correct path, under the same input. Positive = goes further (less cautious). The speed part of a plan change. |
| Clearance | Smallest gap between the planned ego footprint and the altered object over the 3 s: the object's real future boxes for drops, its fixed position for phantoms. |
| Change in clearance | Clearance of the broken path minus clearance of the correct path, under the same input. Negative = comes closer to the altered object. |
| Eligible | A critical drop whose text-only plan change is above the text-only floor (D4): the error changes the plan when the text is all the model has, so with an image there is a genuine conflict. |
| Misled rate | Among eligible items, the share whose real-image plan change is above the real-image floor; likewise with the gray image and the gray-image floor. |
| Expected-direction share | Among items misled with the real image, each component is labelled **expected**, **opposite** or **no clear change**. Expected: for drops, more progress (progress component) and less clearance (clearance component); for phantoms, less progress and more clearance. A change is clear only if its size exceeds the 95th percentile of that component's size over control drops (computed with the floors, D4). Reported as the progress share and the clearance share (expected / opposite / no clear change), plus the share of items where both components are clear and expected. **Robustness:** the sign-only share (every change counted by its sign, with 50% as the no-direction reference) is reported alongside, so the conclusion does not depend on where the threshold sits. |
| Reasoning mentions the altered object | **Exploratory:** checked by hand on a random sample of items (seeded), not automated. |

### How results are averaged and broken down

**Decided:** 2026-11-25

- **Primary: scene-averaged.** Every metric is first averaged within each scene, then across scenes, so each scene counts once. This matches the statistics, which treat the scene as the unit (scene-level bootstrap intervals and the sign-flip test on per-scene means). Scenes contribute 6-8 frames each (D3), so this also removes any weighting from scene length.
- **Secondary: frame-pooled.** Every metric is also reported as a plain mean over all frames, for comparison with LightEMMA and other planning papers, which report this way.
- **If the two differ noticeably,** that is reported and examined rather than ignored.
- **Breakdown by object class (exploratory).** The class of every dropped object is recorded. Results for drop items are reported separately for two groups: vehicles (car, truck, bus) and vulnerable road users (pedestrian, bicycle, motorcycle). This breakdown is exploratory: no hypothesis test is run on it, and groups with few items are reported as counts with that caveat stated. Phantoms are always cars (see the corruption rules), so the breakdown applies to drops only.
- **No class priority** when choosing which object to drop: the selection rule is set in the corruption decision, and class is only used for this breakdown.

### Role of each metric

Only plan change enters the hypothesis tests. A few other things decide **which items enter** them: the eligibility gate (built on the text-only plan change and its floor) and the invalid-output rule (D12). Everything else explains the results: whether the plan steered or braked, came closer or stayed away, how often errors got through, and how well the plans drive. None of it can change whether H1 or H2 is supported.

Not every metric exists for every condition. Metrics on **one path** are computed for every condition, including image only, ego history only and the two simple baselines. Metrics that **compare two paths** (the broken one and the correct one under the same input) exist only for items, three times each: text only, real image and gray image.

**Metrics on one path**

| Metric | Computed for | Role |
| --- | --- | --- |
| L2 vs the driver's path (two conventions, plus the LightEMMA comparison column) | Every condition, including image only, ego history only, constant velocity and constant curvature | Reporting: S1 to S3; broken conditions shown next to their correct-graph versions |
| Collision with real objects | Every condition | Reporting: as for L2 |
| Progress (path length in 3 s) | Every condition | Reporting; feeds the change in progress |
| Clearance to the altered object | The correct and broken paths of each item | Reporting; feeds the change in clearance |
| Invalid-output rate | Every model condition | Reporting, **plus an inclusion rule:** items needing an invalid condition are left out of the plan-change measures (D12) |
| Spread between single runs | Conditions with 3 repeats | Reporting: how much the model varies from run to run |

**Metrics comparing two paths (broken vs correct, same input)**

| Metric | Role |
| --- | --- |
| Plan change, real image | **Test:** H1, H2 |
| Plan change, gray image | **Test:** H2 |
| Plan change, text only | **Gate:** decides eligibility; its size is reported |
| Floors (95th percentile of control drops' plan changes, D4) | **Threshold:** eligibility and the misled rate |
| Misled rate (real image, gray image) | Reporting: the headline result, leads the write-up |
| Heading divergence | Reporting: did the plan steer differently? |
| Change in progress | Reporting: did the plan speed up or brake? |
| Change in clearance | Reporting: did the plan come closer to the object? |
| Expected-direction share | Reporting: do the changes go the way the error points? |

Plan change is also computed between clean conditions for S2 to S4 (for example image only vs image + correct graph); that is reporting too.

**Other**

| Metric | Role |
| --- | --- |
| Reasoning mentions the altered object | Exploratory: a hand-checked sample of items |

### Reporting rules

- **The misled rate leads the write-up**, with the control drops' rate (about 5% by construction) as the reference. It is not tested: it turns each item into yes or no, so it misses partial correction, depends on the floor, and is less sensitive than plan change.
- **The share of ineligible critical drops** is reported as a finding.
- **Signed measures** (change in progress and clearance, signed heading divergence, expected-direction share) are never pooled across drops and phantoms, whose expected directions are opposite.
- **L2 and collision for broken conditions** are shown side by side with the correct-graph condition under the same input (e.g. collision rate with the correct vs the broken graph). No L2 difference is computed: the driver's path measures imitation, not safety, so a broken plan that lands closer to it is not a better plan.

---

## D14. Hypotheses

**Decided:** 2026-11-26 (measure: plan change from each condition's own base, compared with control drops; drops only, phantoms reported separately), 2026-11-27 (Holm confirmed; author's prediction recorded; no minimum meaningful effect).

Both hypotheses concern **eligible critical drops** (D9, D13) on the **primary model, Gemma 4 31B** (D1).

- **H1 (does the error get through?):** a wrong scene graph moves the plan even when the real image is present: the real-image plan change on eligible critical drops is larger than on control drops.
- **H2 (does the image's content correct it?):** the error moves the plan less with the real image than with the gray image, beyond any such difference on control drops.

The per-scene values and the test are in D15. Both hypotheses are tested, with Holm correction (D15).

**Why control drops enter both:** a critical drop's plan change = run-to-run noise + effect of editing the text + effect of the error. A control drop has the first two and close to none of the third, so subtracting it leaves the error's effect (H1). For H2, the control term removes a general effect of the image on how much the model trusts the text, so that what remains is the correction specific to the relevant error.

**Pooling: drops only.** Phantoms are not pooled with drops for the tests: the control items are drops, so they are no placebo for an added object; phantoms push the plan the other way (more cautious), so signed measures would cancel; and they come from a different set of frames (straight, moving). Phantoms are reported separately with the same measures and intervals, with the caveat that their floors come from drops. A **control phantom** (D10), which gives phantoms their own placebo, is optional and exploratory: run only if time allows, and reported as exploratory.

**How outcomes will be read (written before any data):**

| H1 | H2 | Reading |
| --- | --- | --- |
| supported | supported | Errors get through, but the image's content reduces them. |
| supported | not supported | The model follows the text even when the image contradicts it. |
| not supported | supported (items eligible) | The image's content fully corrects the error. |
| not supported | not supported (items eligible) | Any image in the prompt, even an empty one, keeps the error from moving the plan (compare S4). |
| not supported | n/a (most items ineligible) | The model largely ignores the scene graph (see S3). |

**Secondary results** (95% intervals, no hypothesis tests). Each gets the real- and gray-image plan changes, the misled rate, heading divergence, changes in progress and clearance, and the expected-direction share:

- phantoms;
- Gemma 4 E4B;
- the two in-path tags;
- object class (vehicles vs vulnerable road users; exploratory, D13);
- the optional hidden drop: the image cannot show the object, so real- and gray-image plan changes should be about equal; a remaining difference is the image's general effect on trust in the text, an independent check on H2's control adjustment;
- position in the graph text: control-drop plan change for objects early vs late in the list (a critical drop usually removes the first line and renumbers every ID; if position matters, the edit effect differs between critical and control drops);
- H2 without the control adjustment;
- the optional control phantom (D10, exploratory): phantom plan changes compared with control-phantom plan changes;
- the optional closed model, if run.

**Secondary questions** (from the clean conditions in D17, 95% intervals, no hypothesis tests):

- **S1:** Does any model condition beat simple extrapolation (constant velocity, constant curvature)? L2 and collision.
- **S2:** Does the correct graph add anything beyond the image (image only vs image + correct graph)? L2 and collision; plan change between them.
- **S3:** How much of planning is continuing past motion? L2 and collision, ego history only vs the other conditions; plan change of each condition from ego history only.
- **S4:** Does an image slot with no content change behaviour (gray + correct graph vs graph only, correct)? Plan change between them (the floors give the scale); L2.

**Author's prediction before the run** (recorded 2026-11-27, before any val output and before the train-scene pilot):

- **Outcome:** row 1 of the table above: H1 and H2 both supported (errors get through, but the image's content reduces them).
- **Misled rate** (Gemma 4 31B, eligible critical drops): about 30% with the real image, about 80% with the gray image.
- **Share of critical drops that are ineligible:** about 20%.

**Minimum meaningful effect:** not set. There is no established scale for how large a plan change matters, so a number fixed now would be arbitrary; the misled rate and the change in clearance give the effect its practical meaning.

---

## D15. Statistical tests

**Decided:** 2026-11-26, confirmed 2026-11-27 (Holm, the +1 correction and the rerun rule)

- **Unit:** the scene. Item values are averaged within each scene. Scenes without at least one eligible critical drop and one control drop with valid comparisons are excluded and counted.
- **Order:** the floors and eligibility (D4, D13) are computed and recorded before any H1 or H2 value is computed.
- **H1, per scene:** mean real-image plan change over eligible critical drops **minus** mean real-image plan change over control drops.
- **H2, per scene:** mean of (gray-image minus real-image plan change) over eligible critical drops **minus** the same mean over control drops.
- **Test:** two-sided sign-flip permutation test on the per-scene values, 100,000 random sign flips with a fixed seed, alpha = 0.05. p = (count + 1) / (flips + 1), so a p-value is never exactly 0 (Phipson and Smyth, 2010).
- **Rerun rule:** if a p-value lies within 0.002 of the threshold it is compared with (0.025 or 0.05), the test is rerun with 1,000,000 flips and that p-value is used.
- **Two hypotheses: Holm correction.** Both are tested. The smaller p-value is compared with 0.025; if it passes, the other is compared with 0.05; if it fails, neither is significant. This keeps the overall false-positive rate at 5%. It replaces the fixed sequence (H2 only if H1 is significant), which would block H2 exactly when H1 fails because the image fully corrects the error.
- **Effect sizes:** the mean per-scene value for each hypothesis, with a 95% interval from resampling scenes 10,000 times (fixed seed).
- **Robustness (no tests):** the same values using the mean path instead of the median (D4), using per-scene medians of item values instead of means, and H2 without the control adjustment. Any disagreement with the primary result is reported.
- **Everything else** (D14 secondary results and questions) gets 95% intervals and no significance tests.

---

## D16. Budget

**Decided:** 2026-11-27

- **Money cap:** $100 of GPU time on RunPod for the full run, both models, confirmed against the one-scene pilot's measured cost before freezing.
- **Stop date:** the full run stops at the end of 2026-12-02 (author's local time), complete or not.
- **Scene order:** scenes are processed in a random order fixed by seed 0 (the same order as the D2 fallback). If the money cap or the stop date is reached first, the completed scenes are a random subset of the 150; the analyses use them and the number completed is reported.
- **A scene counts as completed** only when every condition for it has finished; partly finished scenes are excluded and counted.
- **Priority:** Gemma 4 31B (primary, the model the hypotheses are tested on) is scheduled first; Gemma 4 E4B runs in the same scene order, in parallel if a second GPU fits within the cap, and may cover fewer scenes, which is reported.
- **If the pilot projects a cost above the cap:** scenes are dropped from the end of the seed-0 order before the run starts (D2), never chosen by hand.

**Why:** a fixed random order means that stopping early removes scenes at random, not the ones that happened to come last in the dataset, so a partial run is still an unbiased sample.

---

## D17. Conditions

**Decided:** 2026-11-25

Every model condition includes the 3 s ego history text. The **gray image** is a uniform mid-gray image (RGB 128, 128, 128) of the same size (1600 x 900), saved as JPEG like the real camera images: it keeps an image slot in the prompt but carries no scene information. (A grayscale copy of the real image would still show the scene, so it would test colour, not content.)

| Grid | Condition | Image | Scene graph | Repeats (D4) | Run on | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | Image only | real | none | 1 | every frame | S2 |
| Clean | Ego history only | none | none | 1 | every frame | S1, S3 |
| Clean | Image + correct graph | real | correct | 3 | every frame | H1, H2 (base of the real-image plan change), S2 |
| Clean | Gray + correct graph | gray | correct | 3 | every frame | H2 (base of the gray-image plan change), S4 |
| Clean | Graph only, correct | none | correct | 3 | every frame | Eligibility (base of the text-only plan change), S4 |
| Clean | Constant velocity, constant curvature | n/a | n/a | n/a (not models) | every frame | S1; constant velocity is also the invalid-output fallback (D12) |
| Corrupted | Image + broken graph | real | one edit | 3 | every item | H1, H2 |
| Corrupted | Gray + broken graph | gray | one edit | 3 | every item | H2 |
| Corrupted | Graph only, broken | none | one edit | 3 | every item | Eligibility |

**Items:** the corrupted conditions run on every item: critical drops (H1, H2), control drops (the placebo: the floors in D4 and the control averages in H1 and H2), phantoms (secondary results), and hidden drops and control phantoms if run (exploratory).

All conditions run on both Gemma models (D1). The optional closed model, if run, uses the same table.

**What each comparison answers:**

| Comparison | Question |
| --- | --- |
| Graph only: broken vs correct | Does the error move the plan when the text is all the model has? (eligibility: is there a genuine conflict?) |
| Image + broken vs image + correct | How much does the error move the plan when the real image could expose it? |
| Gray + broken vs gray + correct | How much does it move the plan with an image slot present but no scene information? |
| Image + broken vs image + correct, critical drops vs control drops | **Does the error get through with the real image?** (H1, D14) |
| The gray- and real-image changes compared, beyond the same comparison on control drops | **Does the image's content correct the error?** (H2, D14) |
| Gray + correct vs graph only, correct | Does an image slot alone, with no content, change behaviour? |
| Image only vs image + correct graph | Does the correct graph add anything beyond the image? |
| Graph only, correct vs image + correct graph | Does the image add anything beyond the correct graph? |
| Clean model conditions vs constant velocity / curvature | Does any input beat simple extrapolation of past motion? |

---

## D18. Records

**Decided:** 2026-11-25

Everything needed to reconstruct every selection and every number is saved, so the method can be checked from the data alone. Raw model responses are never overwritten; if parsing changes, they are re-parsed.

| Record | One row per | Contents |
| --- | --- | --- |
| Frames | evaluated keyframe | scene, sample token, keyframe index, ego speed, path turn over 3 s, manoeuvre label, number of objects in the graph |
| Objects | **every** annotated object in every evaluated frame | class, position, distance, all tags (in view, visible, near, in-path now, entering within 3 s, movable), whether it entered the graph, and **if not, the exclusion reason** (class outside the 10, not in front of the camera, under 50% inside the image, below the height threshold, no LiDAR points) |
| Items | every edit made, **and every edit that could not be made** | kind, frame, object (or phantom position and size), number of candidates, random draw and seed, each phantom check result, or the reason no item was possible |
| Calls | model request | request ID, model and revision, condition, frame, item, repeat, seed, prompt and image hashes, raw response, reasoning text, parsed path, valid or invalid and why, token counts, latency |
| Runs | run of the pipeline | git commit, plan version, vLLM version, Docker image, model revisions, GPU type, start and end time, number of calls |
| Floors | model x input (text-only, real-image, gray-image) x measure (plan change; size of change in progress; size of change in clearance) | floor value (95th percentile over control drops), number of control drops used, time computed (before any H1 or H2 value), code version |
| Item metrics | item x model | the three plan changes, heading divergences, changes in progress and clearance, eligible or not, misled with the real / gray image, expected direction, which conditions were invalid, metrics code version |
| Frame metrics | frame x condition x model | L2 at each 0.5 s step, collision (and at which step), progress, valid or fallback path, metrics code version |

Floors, item metrics and frame metrics are generated from the calls record by versioned code and never edited by hand, so every number in the write-up traces back to a row.

The objects and items records also supply the descriptive statistics for the write-up (for example, the share of annotated objects too small to resolve, and how often each kind of edit was possible).

---

## D19. Checks before the full run

**Decided:** 2026-11-25, 2026-11-26 (tests for the path conversion and the plan-change metrics)

All checks below must pass before the plan is frozen and the full run starts. A failure is fixed in the code (or, before freezing, in this plan) and the affected checks are repeated. This section collects the individual checks listed under D5, D7, D9 and D10.

**1. Automatic tests** (in the repository, run on every code change):

| Test | Passes when |
| --- | --- |
| Driver's real path scored against itself | L2 is 0 at every step |
| L2 on 3 frames computed by hand | the code matches to 1 cm, under both conventions |
| A path driven straight into a known box | collision is flagged |
| Constant velocity on a straight, steady-speed segment | close to the driver's real path |
| Path conversion | constant speed and curvature give points on the exact circle (to 1 mm); zero curvature gives a straight line |
| Median of repeats | taken per coordinate and time step on positions, matching a hand computation |
| Plan change and heading divergence on made-up paths | identical paths give 0; a path shifted 1 m sideways gives a plan change of 1 m and a heading divergence of 0 degrees; a swerve that straightens out gives a non-zero heading divergence (its peak angle); steps under 1 m are skipped |
| Changes in progress and clearance on made-up paths | a longer path gives a positive change in progress; a path closer to the object gives a negative change in clearance |
| Order of computation | the floors and eligibility are written to the records before any H1 or H2 value is computed |
| **No future leak:** future annotations and ego poses altered for a few frames | the model input (image, ego history, scene graph text) is byte-for-byte unchanged |
| Broken graph vs clean graph, every item | differ only in the one edited line and the ID renumbering |
| Drop | the object is absent from the broken graph and present in the clean one |
| Phantom | at least 1 m from every real box, passes the D7 in-view rule, at most 20% hidden behind closer objects, inside the straight current-heading corridor |
| Prompt building | identical inputs produce identical prompt bytes |
| D7 height threshold | one visual token's footprint at budget 1120 is measured from Gemma 4's image processor and recorded in D7 |

**2. Visual review panels.** For a random sample of **30 items** (drawn with a fixed seed, with at least 5 of each item kind: critical drop, control drop, phantom, and hidden drop and control phantom if run) **plus 5 frames with no items**, one panel each:

- **Left:** the front camera image with every box drawn, the dropped object highlighted, and the phantom's projected box.
- **Right:** the bird's-eye view with the driver's 3 s path corridor, the straight current-heading corridor, the camera's view cone, every box coloured by its tags (visible, near, in-path now, entering within 3 s), and the phantom's footprint.
- **Below:** the clean and broken scene-graph text, with the edited line marked.

For each panel, confirm by eye: the graph contains exactly the objects the image could show; the tags look right; the dropped object is clearly visible in the image (or clearly occluded, for a hidden drop); the phantom's spot is visibly empty road; nothing else changed. Each panel's result (pass or fail, with a note) is logged. Any failure: fix, re-run the automatic tests, then review the same 30 panels again plus a fresh 30 drawn with the next seed.

The same panels are reused in the write-up (an example of each kind of edit) and, after the run, for a gallery of failure cases.

**3. Role-line comparison** (settles `prompts/01_role.txt`). LightEMMA's role line ("You are an autonomous driving labeller.") is compared with "You are the planning module of an autonomous vehicle." on **5 train scenes** drawn with a fixed seed (about 35 frames; val stays unseen), in the image + correct graph condition, on Gemma 4 31B, 3 repeats per version. Decision rule, fixed now:

1. If one version's invalid-output rate is lower by at least 5 percentage points, it is chosen.
2. Otherwise, the version whose L2 (average up to 3 s) is lower by more than 10% is chosen.
3. Otherwise, LightEMMA's original line is kept.

About 35 frames can detect only large differences, so rule 3 defaults to the original wording rather than picking whichever came out slightly ahead. L2 measures imitation of the driver, so this is a tie-breaker between wordings, not a search for the "correct" prompt. The choice is made before any val data is seen, so it cannot be tuned towards the hypotheses. Outputs are not used in any analysis.

**4. One-scene pilot.** One scene from the **train** split (not val, so no evaluation data is seen before freezing) is run through every condition on both models. It checks that responses parse into valid paths, and measures token counts, latency, GPU time, cost and the invalid-output rate. It fixes the maximum output length (D1) and confirms the budget (D16). Pilot outputs are not used in any analysis.

**5. Freeze.** When checks 1 to 4 pass, the plan is committed with a dated tag (e.g. `plan-v1`) before the full run starts.

---

_Before freezing: the role-line comparison (D19, step 3), the measured height threshold (D7), the model revisions (D1), the median car size (D10), `scenes.txt` (D2), verification of the KITTI citation (D7), and the maximum output length and cost from the pilot (D1, D16). D11 (noise-floor rule) is covered by D4._
