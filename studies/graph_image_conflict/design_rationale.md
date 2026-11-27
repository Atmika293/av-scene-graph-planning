# Design rationale: does the image correct a wrong scene graph?

**Companion to `analysis_plan.md`.** The plan says *what* will be done; this file records *why*: the question behind each decision, the options considered, what was chosen, what remains risky, and what changed during planning. If the two ever disagree, **the plan wins**. This file is frozen together with the plan (same tag, e.g. `plan-v1`); anything added after the tag is marked as written after freezing.

**Provenance:** working notes drafted with Claude (an AI assistant) during the planning sessions of 2026-11-24 to 2026-11-26, for review by the author. Not for use as application writing.

---

## Contents

1. The question and why it is open
2. What a plan change contains (the effects the design has to separate)
3. Decisions D1 to D19
4. What changed during planning
5. Known limitations
6. Still open

---

## 1. The question and why it is open

**Question:** When a VLM planner's perception text leaves out a pedestrian or vehicle that its camera clearly shows, does it drive as if the object isn't there, or does what it sees rescue the plan?

**Why this wording (revised 2026-11-27).** The original question was: "when a zero-shot VLM planner receives the nuScenes front camera image plus a scene graph, and the scene graph is wrong about one object, does the planned trajectory follow the text or the image?" It asked slightly more than the tests answer, in two ways: "wrong about one object" covers additions (phantoms) as well as omissions, but the confirmatory tests use omissions only; and "follow the text" implies the plan moves towards what the text says, while H1 tests that the plan changes in any direction (direction is reported, not tested). A narrower technical wording ("does the omission change the planned trajectory, and does the image's content reduce that change?") matched the tests but read like a specification and undersold the question. The chosen wording stays plain and scoped to omissions, and puts H2's question, whether seeing rescues the plan, first. The precise scope is stated under it in the plan.

**Why it is open:** prior work studies the pieces separately. Structured scene descriptions are used as planner input (e.g. GraphPilot), and modality conflict between image and text is studied in question answering (e.g. Deng et al., "Words or Vision"; AutoTrust's misinformation probe, the closest prior work, uses free-text false claims in driving QA). No study found during planning gives a driving VLM *planner* structured perception text that contradicts the image and measures what happens to the *trajectory*. A planner turns a wrong perception into a motion, so the answer matters for any system that feeds detector output to a language-model planner.

**What would count as an answer:** whether a single wrong object in the scene graph moves the plan when the real image is present (H1), and whether the image's content, not just the presence of an image, reduces that effect (H2).

---

## 2. What a plan change contains

The whole design follows from one observation: **a plan change is never just the effect of the error.** When the correct scene graph is replaced by a broken one and the two plans are compared, the difference contains several parts, and each needs something in the design to separate it out.

### The parts

| Part | Where it comes from | What handles it | Hypotheses that depend on it |
| --- | --- | --- | --- |
| **Run-to-run noise** | Sampling at temperature 1.0: identical requests give different plans | 3 repeats combined by their median (reduces it); control drops (subtracted in H1 and H2; set the floors) | H1, H2, eligibility, misled rate |
| **Effect of editing the text at all** | A broken graph is a line shorter, IDs are renumbered, the model reasons through a slightly different prompt | Control drops: the same procedure applied to an object that should not matter (a placebo) | H1, H2 |
| **Different starting plans** | Different inputs give different plans even with a correct graph (an image may show the road curving) | Each plan change is measured from the correct-graph plan *under the same input* | H1, H2, eligibility |
| **Effect of an image slot being present** | Any image in the prompt, even an empty one, may change how the model plans | The gray image (uniform mid-gray, no scene content); S4 | H2 |
| **General effect of image content on trust in the text** | With real content to look at, the model may lean less on the text for *any* edit, relevant or not | H2's control adjustment; the hidden drop as an independent check; H2 unadjusted also reported | H2 |
| **The error's own effect** | What we want to measure | Critical drop minus control drop (H1); gray minus real, beyond the same on control drops (H2) | H1, H2 |

### How the parts combine

For a single item under one input:

> critical drop's plan change = noise + edit effect + **error effect**
> control drop's plan change = noise + edit effect + (close to 0)

So **H1**, per scene, is

> (average real-image plan change on eligible critical drops) - (average on control drops) = error effect with the real image

and **H2**, per scene, is

> (gray - real, averaged on eligible critical drops) - (gray - real, averaged on control drops)

The first part of H2 is how much less the error moves the plan when the image shows the scene. The control part removes the general effect of image content on trust in the text, leaving the correction that is specific to the relevant error: the object in the image contradicts the text.

### Other threats the design guards against

| Threat | Guard |
| --- | --- |
| **Selecting items by their outcome** | Eligibility is decided on graph-only runs, which are separate from the image and gray runs the tests measure. Floors and eligibility are computed and recorded before any H1 or H2 value. Frames are chosen blind to content; random draws are seeded. |
| **Future information leaking into the input** | The scene graph uses the current and previous keyframe only; "ahead in our direction" uses the current heading, not the future path; phantoms are placed only on straight frames, where straight ahead reveals nothing; nuScenes' parked/stopped attributes are not used. An automatic test alters future annotations and checks the model input is byte-for-byte unchanged. |
| **The graph and the image disagreeing for reasons other than the edit** | The graph includes exactly the objects the front camera could show (D7 in-view rule), with no cap on count, so the only deliberate mismatch is the edit. |
| **Text giving away the edit** | After every edit the graph is re-sorted and IDs reassigned; a phantom's line is produced by the same code as real objects. |
| **Metrics that can hide a response** | Plan change counts any change; its direction and speed parts (heading divergence, change in progress) show what kind of change happened, so a braking-only response is not missed. |

---

## 3. Decisions D1 to D19

Numbers follow the Phase 0 list. D11 (noise-floor rule) is covered by D4.

### D1. Models

- **Question:** which model(s), and how to run them so the result is verifiable?
- **Considered:** Claude Haiku (cheap, but an older generation that might make the study feel stale); Claude Sonnet (current, moderately priced); open-weight models through pay-per-token providers; open-weight models self-served.
- **Chosen:** Gemma 4 31B (primary) and Gemma 4 E4B (secondary), official BF16 weights, served with vLLM on RunPod. Claude Sonnet 5.5 kept only as an optional exploratory run.
- **Why:** open weights make the study verifiable and permanently rerunnable, and allow later inspection of internals (attention to image vs graph tokens). The same family at two sizes isolates size while keeping training and prompt format; E4B is small enough for edge devices, which makes the comparison practically meaningful. Pay-per-token providers were rejected because none guaranteed BF16 for Gemma 4 31B (several serve FP8 or FP4), so the precision tested could not be stated. A closed reference model was judged to add little; it is pre-listed as optional so that adding it later is not a choice made after seeing results.
- **Settings:** Google's recommended sampling (temperature 1.0, top-p 0.95, top-k 64), because the study should test the model as it is meant to be used; reasoning on, because a planner is expected to reason; the highest image budget (1120 tokens), so small objects keep detail; image first, as Google recommends.
- **Risks:** the vision encoders differ in size (about 550M vs 150M), so 31B vs E4B is a comparison of whole models, not of language-model size alone. Results apply to Gemma 4 only.

### D2. Scenes

- **Question:** how many scenes, from which split?
- **Considered:** the test split (no public annotations); a subset of 40 val scenes (cheaper); all 150 val scenes.
- **Chosen:** all 150 val scenes.
- **Why:** the scene is the unit of analysis, so the number of scenes sets the precision; all 150 needs no sampling choice to defend and matches LightEMMA's evaluation set. The test split was rejected because the scene graph, visibility tags and collision check need 3D annotations. Train is kept untouched for the later detector study.
- **Risks:** cost. A seeded rule (drop scenes in a fixed random order) handles going over budget without judgement.

### D3. Frames

- **Question:** which keyframes to evaluate?
- **Considered:** every 2nd keyframe (about 2,100 frames, double the cost), every 4th (one per 2 s), every 6th (fully independent future windows, only 4-5 per scene); a cap of 7 per scene vs no cap.
- **Chosen:** every 4th eligible keyframe, starting at the first with 3 s of history and 3 s of future, no cap per scene.
- **Why:** neighbouring keyframes 0.5 s apart are near-duplicates; variety across scenes matters more than frames per scene. A cap was first set at 7, then removed: there is no reason to discard valid frames, and scene-level averaging already stops long scenes from weighing more. History and future are still read at 0.5 s from all keyframes; the spacing only decides which frames are evaluated.
- **History length (revised 2026-11-27):** LightEMMA's code computes 6 speeds and 6 curvatures from 8 poses, because each curvature needs a pose on either side, so a frame needs 7 keyframes of history, not 6. The first evaluated keyframe moved from 6 to 7.

### D4. Repeats and noise floor

- **Question:** how many runs per condition, how to combine them, and what counts as "more than noise"?
- **Considered:** 1, 2 or 3 repeats; combining by median or mean, over positions or over raw speed and curvature; a floor from pairs of single repeats, from 6 repeats of the baseline, or from control drops.
- **Chosen:** 3 repeats for compared conditions (1 for image only and ego history only, which are never compared frame by frame); the median path per coordinate and time step, over positions, with the mean as a check; floors as the 95th percentile of each plan change over control drops.
- **Why 3 repeats:** a third run acts as a tie-breaker, so a median is robust to one odd output, and gives a better estimate of variation than 2.
- **Why medians over positions:** the median of speeds and curvatures, once converted, can be a path no run actually planned.
- **Why the floor comes from control drops:** a plan change compares a median of 3 with a median of 3. Pairs of single repeats differ more than medians do, so a floor from them is too high; getting a median-vs-median floor from the baseline alone would need a second set of 3 runs (6 per condition). Single repeats also never change the text, so they miss the edit effect. A control drop is measured exactly like a critical drop, includes the edit effect, and costs no extra calls.
- **Risks:** if a control object matters slightly, the floor is slightly high (the cautious direction). One floor per comparison treats slow and fast frames alike, although paths spread more at speed.

### D5. In-path

- **Question:** which objects are in the car's path, for choosing critical and control drops?
- **Considered:** in-path now only; now or within the 3 s horizon; corridor width; a map-based lane definition.
- **Chosen:** a 2 m-per-side corridor around the driver's real path over 3 s, extended to the near distance; in-path if the object overlaps it now or at any of the next 6 keyframes; two tags kept separately (in-path now, entering within 3 s).
- **Why:** an object about to enter the path (a pedestrian stepping off the kerb) is exactly the kind whose omission matters. The label uses hindsight, but it is used only to select and group items, never shown to the model, so nothing leaks. Keeping two tags lets the foreseeable and the unforeseeable cases be compared.
- **Risks:** the corridor follows what the driver did, so an object the driver steered around may fall outside it, and timing is ignored. A map-based lane is a next step.

### D6. Near

- **Question:** how close is "near"?
- **Considered:** a fixed distance; a distance tied to the planning horizon.
- **Chosen:** max(15 m, 3 s x current speed), from the ego centre to the nearest point of the object's footprint, with speed from the current and previous poses only.
- **Why:** the model plans 3 s ahead, so anything reachable within 3 s can affect the plan; a fixed distance is too generous in slow traffic and too short at speed. The 15 m minimum keeps nearby objects relevant when stopped.

### D7. In view and visible

- **Question:** what could the front camera show, and what does it actually show?
- **Considered:** a distance cutoff (50 m); a minimum projected box area; a minimum projected box height tied to the model's image resolution.
- **Chosen:** in view = all corners in front of the camera, at least 50% of the projected box inside the image, and the visible part at least one visual token tall at budget 1120 (about 36 px, to be measured). Visible = in view, nuScenes visibility of 40% or more, at least one LiDAR point.
- **Why:** whether a model can see an object depends on its size in the image, not its distance, so a size threshold is more defensible than an arbitrary distance. Tying it to one visual token's footprint links it to how the model actually sees the image (the threshold first used Claude's 28 px patch, then changed with the switch to Gemma). Height, not area, so thin pedestrians are not penalised. The graph must never describe what the image cannot show, or graph-only conditions would have an information advantage. Visibility separates objects the image could correct from those it could not (the hidden drop).

### D8. Scene-graph contents

- **Question:** what goes in the scene graph, and how is it written?
- **Considered:** capping the number of objects; using nuScenes' labelled attributes; object-to-object relations.
- **Chosen:** the 10 detection classes, in view and with at least one LiDAR point; no cap; class, position, distance, heading, speed, moving or stopped; ego-centric relations only; one text line per object, nearest first.
- **Why:** the graph imitates a camera-plus-LiDAR perception system limited to the front view, as a typical scene graph built from detections would. A cap would make the graph omit objects the image shows, a mismatch unrelated to the deliberate errors. nuScenes' parked/stopped attributes were set by annotators who saw the whole sequence, so they carry hindsight. Ego-centric relations mean removing one line removes everything about that object.

### D9. Drop rules

- **Question:** which object to remove, and how to check that removing *any* object is not what drives the effect?
- **Considered:** cycling through every possible object per frame; one per kind per frame; several errors in one item; a priority order of classes.
- **Chosen:** at most one item of each kind per frame, each with exactly one edit: critical drop (nearest movable, visible, near, in-path object), control drop (random movable, visible, far or off-path object), optional hidden drop (nearest near, in-path object that is mostly occluded). No class priority.
- **Why:** one edit per item keeps every effect traceable. The nearest critical object is the most dangerous omission and is chosen by rule, not judgement. The control drop is the placebo that later became central to H1, H2 and the floors. The hidden drop separates genuine looking from general distrust of text: the image cannot show the object, so it cannot correct the error. Class is recorded for an exploratory breakdown rather than used to choose.

### D10. Phantom rule

- **Question:** how to add a non-existent object that the image clearly contradicts, without leaking the future?
- **Chosen:** a stationary median-sized car straight ahead at 10 to min(25 m, near) m, only when moving faster than 2 m/s on a nearly straight path; at least 1 m from every real box; in view; at most 20% covered in the image by closer objects; no redraws if a check fails.
- **Why:** placing it straight ahead on straight frames means its position says nothing about where the car will go. The checks make it checkably wrong: the image must clearly show the spot is empty. No redraws, so the frames with phantoms are not chosen by trial and error.
- **Risks:** phantoms come from a different set of frames than drops (straight, moving), which is one reason they are not pooled with drops for the tests.
- **Optional control phantom (2026-11-27):** the same car on the same frames and at the same distance, shifted 5 to 8 m sideways outside the path corridor: an irrelevant false addition, the placebo for phantoms. Optional and exploratory because it costs about 10 to 15% more calls on a tight timeline and does not affect H1 or H2; its rule is written now so that, if run, it was not designed after seeing results.

### D12. Output format and invalid outputs

- **Question:** what does the model output, how is it turned into a path, and what happens when the output is unusable?
- **Considered:** waypoints (x, y) vs (speed, curvature) pairs; constrained decoding; retries; dropping invalid frames vs a fallback path; a forward-step conversion vs an exact arc.
- **Chosen:** 6 (speed, curvature) pairs as a list, `[(v1, c1), ...]`, positive curvature = left, matching the ego-history format and LightEMMA's output format (JSON was planned until the LightEMMA check showed their format); exact-arc conversion into 6 positions in a frame frozen at the current moment; no constrained decoding; no retries; for L2 and collision, an invalid condition is scored as constant velocity, with valid-only results alongside; for plan changes, items needing an invalid condition are excluded and counted.
- **Why:** no constrained decoding makes the invalid-output rate a genuine measurement. Without a fallback, failures would disappear and the model would be scored only on the frames it handled, often the easier ones. A fallback path cannot be used for plan changes, because it would create a fake change. The exact arc involves no approximation; a forward step, which is what LightEMMA uses, ends almost 1 m off sideways after 3 s on a gentle curve. LightEMMA's conversion and L2 averaging are kept only as a comparison column, so our numbers can be set against theirs.

### D13. Metrics

- **Question:** how to measure whether the plan follows the text or the image?
- **Considered:** a text-following ratio (projecting the image condition's change onto the text-only change); an off-direction change to catch what the ratio missed; direction only (heading divergence); a misled rate; plan change measured from each condition's own base.
- **Chosen:** plan change (mean distance over the 6 positions) as the tested measure, measured from the correct-graph plan under the same input; heading divergence and change in progress as its direction and speed parts; change in clearance as the safety view; the misled rate as the headline description.
- **Why the ratio was dropped:** the two changes it compared started from different plans (graph only vs image + correct), so the angle between them mixed two different reactions; summing products over cumulative positions counted early differences again at every later step; and it was unbounded, needing extra rules to read.
- **Why heading divergence uses the largest angle (revised 2026-11-27):** the peak swerve is what a reader wants to know ("the plans diverged by up to 5 degrees"), and straight steps do not dilute it (in a swerve that straightens out, the mean is 2.3 degrees and the maximum 5). The maximum is noisier and grows with the number of usable steps, so steps under 1 m (the noisiest directions) are skipped, the mean is recorded too, and critical drops are always shown next to control drops computed the same way. It has no floor: it is a reporting measure that explains how the plan changed, and nothing is decided by it.
- **Why the expected-direction share counts only clear changes (revised 2026-11-27):** the share is built from the signs of the changes in progress and clearance; it says how consistently the changes go the way the text points, which an average cannot (a few large changes can pull an average one way while most items go the other). Counting signs alone would let tiny, noise-sized changes count as "expected", so a component counts only if its size exceeds what control drops typically show (their 95th percentile), and the result is reported as expected / opposite / no clear change, separately for progress and clearance.
- **Justification of the noise threshold:** (1) some change in progress and clearance happens with no relevant error at all, because two medians of 3 noisy runs always differ a little; that level has to be measured, not assumed. (2) Control drops measure it directly: they are placebo edits measured exactly like critical drops (median against median, same input), so their changes show how large a change arises from noise plus the effect of editing the text. (3) The 95th percentile is the cutoff: a change larger than 95% of placebo changes is unlikely to be noise; this is the same convention as the floors (D4) and matches alpha = 0.05. (4) It is applied per component because a plan can change in one way only: an item misled by steering may have a noise-sized progress change, whose sign would add a coin flip to the share. (5) It is computed automatically with the floors, per model and input, before any H1 or H2 value, and recorded. **Caveats:** if control objects matter slightly, the threshold is a little high and more items fall under "no clear change" (the cautious direction); the clearance threshold comes from far or off-path objects while critical objects are near and in-path, so it is a rougher scale than the progress threshold, which does not depend on the object; 95% is a convention. **Robustness:** the sign-only share (50% = no direction) is reported alongside; noise pulls it towards 50%, never away, so it is cautious, and if both versions agree the conclusion does not depend on the threshold.
- **Why not direction alone:** the most common reaction to a dropped pedestrian is speed, not steering (the correct plan brakes, the broken one drives on), so heading divergence would be near zero while the plan clearly changed.
- **Why the misled rate is not the test:** it misses partial correction (3 m with gray, 1 m with real counts as misled both times), depends on where the floor sits, and is less sensitive. It is easy to explain, so it leads the write-up, as the planning counterpart of the "Words or Vision" measure.
- **Why both L2 conventions:** the field uses two (average up to t, error at t), and L2 is not part of the tests, so neither needs to be primary.
- **Averaging:** per scene, then across scenes (frames within a scene are not independent; matches the tests); frame-pooled as well, for comparison with planning papers.

### D14. Hypotheses

- **Question:** what exactly is tested, and on which items?
- **Chosen:** H1 (the error gets through with the real image) and H2 (the image's content reduces it), on eligible critical drops, Gemma 4 31B. Drops only; phantoms reported separately.
- **Why these two:** H1 alone cannot say whether the image did anything; H2 alone cannot say whether there was anything to correct. Together they distinguish "follows the text", "image partly corrects", "image fully corrects", "any image suppresses the error" and "ignores the graph" (outcome table in the plan).
- **Why control-adjust H2:** without real content, the model may lean on the text more, so even irrelevant edits may move gray-image plans more. Unadjusted, H2 could be positive because the image lowers trust in the text in general, which is a different finding from correcting a specific error.
- **Why eligibility:** items whose error changes nothing even with text alone are not conflicts; they would add near-zero values that say nothing about the image.
- **Why no minimum meaningful effect:** there is no established scale for how large a plan change matters, so any threshold fixed now would be arbitrary; the misled rate and the change in clearance give the effect its practical meaning.
- **Why drops only:** the control items are drops, so they are no placebo for an added object; phantoms push the plan the other way, so signed measures would cancel; and phantoms come from different frames.

### D15. Statistical tests

- **Question:** how to decide whether H1 and H2 are supported, and how to handle testing two hypotheses?
- **Considered:** a paired t-test or Wilcoxon signed-rank test on the per-scene values; a sign-flip permutation test; testing on frames or items instead of scenes; Bonferroni, a fixed sequence or Holm for the two hypotheses; enumerating every sign pattern vs sampling them.
- **Chosen (confirmed 2026-11-27):** one value per scene for each hypothesis; a two-sided sign-flip permutation test with 100,000 random sign patterns (fixed seed), p = (count + 1) / (flips + 1); alpha = 0.05; Holm correction across H1 and H2; 95% intervals from resampling scenes 10,000 times.

**The logic of a test, briefly.** The null hypothesis is "no effect": for H1, critical drops move the plan no more than control drops. The p-value is the probability, *if the null were true*, of a result at least as extreme as the one observed. A small p means chance alone rarely produces such a result, so the null is rejected and the hypothesis is supported. It is not the probability that the null is true, and it says nothing about the size of the effect (that is the interval's job). A large p does not show there is no effect, only that the data cannot tell an effect from noise.

**Why one value per scene.** Frames from the same scene share the road, traffic and weather, so they are not independent; treating them as independent would overstate the evidence. The scene is the unit in the averaging (D13), the intervals and the test.

**Why the sign-flip test.** If the null is true, each scene's value is just noise, equally likely to be positive or negative, so the observed sizes could have come with any pattern of signs. Flipping the signs at random and recomputing the average shows how large an average chance produces for these particular scenes; the p-value is the share of random averages at least as far from zero as the real one. It goes back to R. A. Fisher's randomisation reasoning in the 1930s, is the randomisation counterpart of the paired t-test and the Wilcoxon signed-rank test, and is widely used (for example in brain imaging). It was chosen over the t-test because it assumes nothing about the shape of the distribution: plan changes are distances, likely skewed with a long tail, not bell-shaped. Its assumptions are that scenes are independent and that, under the null, each value is symmetric around zero; for H1 that holds as long as critical and control drops would behave alike when the error does not matter, which is what the control drop is designed for.

**Why two-sided.** A random average counts if it is at least as far from zero in either direction. H2 in particular could come out negative (the real image making errors worse), and a one-sided test would be blind to that.

**Why alpha = 0.05.** Alpha is the false-positive rate accepted in advance: when there is truly no effect, the test wrongly declares one 5% of the time. 0.05 is a convention (going back to Fisher), not a mathematical property; a lower alpha means fewer false positives but more real effects missed. Other fields choose differently (some researchers argue for 0.005; particle physics requires about 1 in 3.5 million). What matters is that it is fixed before any data are seen, so the threshold cannot be chosen to fit the result. Exact p-values are always reported, because p = 0.049 and p = 0.0001 are very different strengths of evidence.

**Why 100,000 random sign patterns instead of all 2^150.** With about 150 scenes there are 2^150 (about 1.4 x 10^45) sign patterns, far too many to compute; with 5 scenes all 32 could be listed. A random sample of patterns estimates the exact p-value the way a large poll estimates a population. Each random pattern is a trial whose "success" is an average at least as extreme as the real one, with success probability equal to the exact p-value, so the count follows a binomial distribution and the estimate's margin is about 2 x sqrt(p (1 - p) / N). At p = 0.05 and N = 100,000 that is about +/-0.0014; at Holm's first threshold (0.025) about +/-0.001. This margin reflects only which patterns were drawn, not uncertainty in the data. It matters only if p lands within that margin of a threshold; in that case the rule (proposed) is to rerun with 1,000,000 flips, which takes seconds. The fixed seed means anyone rerunning gets exactly the same p-value.

**Why the +1 correction (proposed).** The real sign pattern is itself one of the possible patterns, so it is counted once: p = (count + 1) / (flips + 1). This also means a p-value can never be exactly 0, which would claim more certainty than a sample can give. Without it, sampled permutation p-values are understated by about 1 divided by the number of permutations (Phipson and Smyth, 2010).

**Why 10,000 resamples for the intervals.** The interval is built by drawing the scenes with replacement 10,000 times and recomputing the average each time; the middle 95% of those averages is the interval. 10,000 is the usual standard and makes the interval's ends stable. The 95% level matches alpha = 0.05. Intervals are reported per hypothesis and are not adjusted for multiple testing, which is normal practice; the correction applies to the yes/no decisions.

**What a 95% interval is.** A range of plausible values for the true average, built by a procedure that captures the true value in 95% of repeated studies. Here it comes from resampling scenes (above).

**Why intervals, not tests, for the secondary results and questions.** A test gives yes or no; an interval gives the size of the effect and how precisely it is known, so the reader can judge whether it matters. For example, for S2 (does the correct graph improve L2?), with the value = L2 of image only minus L2 of image + correct graph:

| Interval | Reading |
| --- | --- |
| 0.12 m [0.05, 0.19] | The graph helps a little, and reliably |
| 0.12 m [-0.08, 0.30] | Cannot tell; the data are too noisy |
| 0.01 m [-0.02, 0.03] | The graph makes essentially no difference: a narrow interval around 0 is an informative result, which a test cannot give |
| 0.80 m [0.60, 1.00] | The graph clearly matters, and by a lot |

(Illustrative numbers.) There are also two practical reasons. Every extra test would make Holm stricter on H1 and H2, weakening the main question to test side questions. And with many secondary results, about 1 in 20 intervals would exclude 0 by chance even with no real effect, so treating each as a yes/no claim would produce false findings.

**What this implies in practice.** H1 and H2 answer the main question with a controlled error rate and carry the only confirmatory claims. The intervals on everything else give context (how big, how consistent, how precisely known) and are written as descriptive ("consistent with", "suggests"), never as confirmed findings. An interval that excludes 0 roughly corresponds to p < 0.05, but without a correction for the number of results, so it is not read as one.

**Why a correction for two hypotheses.** Testing two hypotheses at 5% each gives up to about a 10% chance that at least one is a false positive (the family-wise error rate). A correction brings it back to 5% for both together.

**Why Holm, and how it works.** Bonferroni tests each at alpha divided by the number of tests (0.025 each); it is safe because the chance of any false positive is at most the sum of the individual chances, but it is strict. Holm (1979) sorts the p-values: the smallest must be below 0.025; if it passes, the next must be below 0.05; at the first failure, it and everything after it are not significant. It still keeps the family-wise error rate at 5% for any combination of true and false hypotheses, with no assumption about how the tests relate: a false positive requires rejecting the true-null hypothesis with the smallest p-value, which is compared with a threshold of at most alpha divided by the number of true nulls, and by the same sum argument as Bonferroni that happens with probability at most alpha. Holm is never worse than Bonferroni (same first threshold, looser second one).

**Why Holm over the fixed sequence.** The fixed sequence gives H1 the full 0.05 and tests H2 only if H1 passes. That is also valid, but with the plan-change measure H1 can fail because the image fully corrects the error, which is exactly when H2 matters most, and the fixed sequence would block H2 there. Holm costs some strictness on the first test (0.025 instead of 0.05) but lets either hypothesis come out significant. Example: p(H1) = 0.30 and p(H2) = 0.01 gives H2 significant under Holm and nothing under the fixed sequence; p = 0.03 and 0.04 gives neither under Holm, the price of testing two things.

- **Risks:** the test treats scenes as independent; the symmetry assumption for H2's control-adjusted value relies on critical and control drops behaving alike under the null; with fewer usable scenes than expected (many excluded for lacking items), power drops, which is reported.

### D16. Budget

- **Chosen:** $100 of GPU time; the full run stops at the end of 2026-12-02; scenes processed in a random order fixed by seed 0; a scene counts only when all its conditions are done; Gemma 4 31B scheduled first.
- **Why:** the stop date leaves time for analysis and writing before the 2026-12-05 target. A fixed random order means that stopping early removes scenes at random, so a partial run is still an unbiased sample. The primary model goes first because the hypotheses are tested on it.

### D17. Conditions

- **Question:** which input combinations are needed?
- **Considered:** a grayscale copy of the real image vs a uniform gray image; combining the gray image with graph errors.
- **Chosen:** clean conditions on every frame (image only, ego history only, image + correct, gray + correct, graph only + correct, constant velocity and curvature); corrupted conditions on every item (image, gray and graph only, each with the broken graph).
- **Why a uniform gray image:** a grayscale copy still shows the scene, so it would test colour, not content. The uniform gray image keeps an image slot with no scene information, separating "an image is present" from "the image shows the scene".
- **Why the gray image is not a second error:** every gray-image comparison is made against gray + correct, so within the comparison the graph edit is still the only difference. Gray + correct was initially missing and was added for this reason.

### D18. Records

- **Why:** every selection and every number must be reconstructable from the data alone. Records include the objects left out of the graph and why, and the edits that could not be made, because those supply descriptive results and expose selection effects. Metrics and floors are generated from the calls by versioned code, never edited by hand.

### D19. Checks before the full run

- **Why:** most errors in this kind of study are silent (a wrong sign, a leak, a misprojected box). Automatic tests catch the computable ones, including a no-future-leak test and tests on made-up paths with known answers. Visual panels catch what tests cannot (a graph that does not match the image). The pilot runs on a train scene so no val data is seen before freezing.
- **Why a role-line comparison:** LightEMMA's role line ("autonomous driving labeller") fits a planner poorly, but it is their tested wording. Rather than choose by taste, the two lines are compared on train scenes before freezing, with a decision rule fixed in advance that defaults to LightEMMA's line unless the alternative is clearly better (invalid rate at least 5 points lower, or L2 more than 10% lower). It uses train data only and a measure outside the tests, so it cannot steer the hypotheses.

---

## 4. What changed during planning

| Topic | From | To | Why |
| --- | --- | --- | --- |
| Models | Claude (Haiku or Sonnet) | Gemma 4 31B and E4B, open weights, BF16, self-served | Verifiability, access to internals, a same-family size comparison; providers did not guarantee precision |
| Closed reference | Sonnet as a reference for critical items | Optional exploratory run only | Little added value for the question |
| Scenes | 40 val scenes | All 150 | Precision at the scene level; no sampling choice to defend; matches LightEMMA |
| Frames per scene | Cap of 7 | No cap | No reason to discard valid frames; scene averaging already balances scenes |
| Repeats | 2 | 3 | A tie-breaker; a better noise estimate |
| Graph inclusion | Within 50 m | Projected box height of at least one visual token | Visibility depends on size in the image, not distance |
| Height threshold | Claude's 28 px patch | Gemma 4's visual-token footprint (about 36 px) | Followed the model switch |
| Gray condition | Grayscale copy considered | Uniform mid-gray | A grayscale copy still shows the scene |
| Baselines | No gray + correct | Gray + correct added | Needed as the base for gray-image comparisons |
| Path conversion | Forward step | Exact arc | About 1 m of sideways error after 3 s |
| Main measure | Text-following ratio, plus off-direction change | Plan change from each condition's own base, with heading divergence and change in progress as parts | Different base points; double counting over positions; unbounded |
| Test measure | Misled rate considered | Plan change tested; misled rate leads the write-up | Partial correction, threshold dependence, sensitivity |
| Noise floor | 95th percentile of single-repeat pairs | 95th percentile of control drops' plan changes | Median vs median; includes the edit effect; no extra calls |
| H1 | Ratio above zero | Critical drops above control drops | Plan change is never exactly zero |
| H2 | Gray minus real | Gray minus real, beyond the same on control drops | Separates specific correction from a general drop in trust |
| Multiple testing | Fixed sequence | Holm (confirmed 2026-11-27) | H2 stays testable when the image fully corrects the error |
| History length | 6 keyframes before | 7 keyframes before | LightEMMA's curvature needs a pose on either side |
| Heading divergence | Mean angle over steps; floor from control drops; steps under 0.5 m skipped | Largest angle (mean recorded); no floor, critical and control side by side; steps under 1 m skipped | The peak swerve is the useful number; it is a reporting measure, so it needs a scale, not a cutoff |
| Expected-direction share | Sign of progress or clearance change, counted under both if they disagree | Expected / opposite / no clear change per component, clear only beyond control drops' 95th percentile | Tiny changes should not count by their sign alone; clearer reporting |
| Prompt role line | LightEMMA's line, by default | Chosen by a pre-registered comparison on train scenes | Avoids choosing by taste; defaults to LightEMMA unless clearly worse |
| Output format | JSON | LightEMMA's list of pairs | The plan said the format follows LightEMMA |
| Control phantom | To be decided later | Optional, exploratory, rule written now | Cheap insurance against designing it after seeing results |
| Pooling | Drops and phantoms pooled | Drops only; phantoms separately | The control is a drop; opposite expected directions; different frames |
| Research question | "Wrong about one object... follow the text or the image?" | "Leaves out an object the camera clearly shows... drive as if it isn't there, or does what it sees rescue the plan?" | Matches the confirmatory scope (omissions; change, not direction) while keeping a plain-language question |
| L2 change (broken minus correct) | Proposed as a reported measure | Dropped; L2 and collision shown side by side instead | L2 measures imitation of the driver, not safety; the numbers were already reported |

---

## 5. Known limitations

- **Ground-truth scene graph.** Apart from the one edit, the graph is perfect. Real detector output has many small errors at once; studying that is a later step (the detector study).
- **Front camera only, open loop.** The plan is never executed, so the effect of an error on what happens next is not measured. Open-loop L2 is known to reward continuing past motion (the ego-status critique); S1 and S3 report how much of planning is extrapolation.
- **Zero-shot, two models from one family.** Results apply to Gemma 4 31B and E4B, with the stated prompts and settings.
- **Hindsight in the in-path tag** ("entering within 3 s"), used for selection only; results are split by tag.
- **The corridor follows the driver's path,** so objects the driver steered around may be missed, and timing is ignored.
- **Control drops may have a small real effect,** which makes H1 and the floors slightly cautious.
- **The edit effect may differ by position in the list** (critical drops usually remove the first line); checked with the list-position comparison.
- **Phantoms use floors from drops** until a control phantom is decided.
- **One floor per comparison** for slow and fast frames alike.
- **Heading divergence skips near-stationary steps,** so it is undefined for plans that stop; change in progress covers those.
- **Path conversion and history format** depend on the LightEMMA check.

---

## 6. Still open

- The prompt's role line (`prompts/01_role.txt`).
- Before freezing: the measured height threshold (D7), the model revisions (D1), the median car size (D10), `scenes.txt` (D2), verification of the KITTI citation (D7), and the maximum output length and cost from the pilot (D1, D16).
