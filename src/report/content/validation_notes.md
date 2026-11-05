<!--
DRAFT (Claude): taken from the validation scripts' own docstrings; trim or
reword as you like.
-->
## Design notes

**Separate checks for separate failure modes.** The table-row check runs its column-set
comparison and its value comparison independently, so a row with the wrong *shape* can't hide
a row with the wrong *values*, or the other way round.

**Localise, don't just detect.** The projection check doesn't only compare final 2D boxes;
it compares corners after every transform stage against a fork of nuScenes' own exporter. An
end-to-end mismatch says *something* is wrong; a stage-by-stage diff says *which transform*.

**A physical fact can be a reference.** The orientation check doesn't assume the ego frame's
"+x is forward" convention to test it: it uses the fact that a car moves mostly forward between
two samples half a second apart. A swapped or flipped axis would show up as sideways or
backward motion. Scenes where the car never moves are skipped, since there's no heading to
measure.

**The strongest check uses someone else's numbers.** nuScenes records how many LiDAR points
fall inside each annotated box. Recomputing that count from my own ego-frame points and boxes
(using the same point-in-box algorithm nuScenes used) and matching it exactly means the LiDAR
loading, both pose transforms and the box geometry all agree with the dataset's own.

**Not everything gets a pass/fail.** The box-separation ratio has no ground truth: a low value
can mean a misaligned box, or two real objects parked bumper-to-bumper. It's reported as a
diagnostic, and its lowest cases (mostly barriers and pedestrians, which do stand close
together) are flagged for inspection rather than failed.

## Limits

- **Projection is checked against the devkit's implementation, not an independent ground
  truth.** nuScenes has no separately annotated 2D boxes, so agreeing with the devkit is the
  strongest claim available.
- **Boxes with no LiDAR points are excluded** from the projection comparison
  ({{stat:projection.skipped_no_lidar|,}} annotations): the pipeline only builds annotations
  with at least one LiDAR point, so those never enter the comparison.
- **Visual views have no score.** The BEV renders are checked through the data that feeds them
  (every row above); the rendering itself was checked by eye, which is how the image-axis flip
  described in the [build log](build.html) was caught.
