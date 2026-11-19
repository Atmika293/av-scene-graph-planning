[← All papers](papers.html)
{: .meta }

# Lift-Splat-Shoot

Philion & Fidler · [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf) · <span class="badge progress">Reading</span> · read for [Phase 2](learning.html)
{: .meta }

How to build a bird's-eye view from cameras alone, where depth has to be predicted rather
than measured. My Phase 2 BEV does the same job geometrically, with LiDAR supplying every
point's position, so most of my notes compare the two.
{: .lede }

<!-- DRAFT (Claude): the Summary section is a placeholder. Rewrite it in your own words; "My notes" below is yours as written. -->
## Summary

- **Problem:** turn the images from a car's cameras into one top-down map (for example, which
  cells are vehicle or drivable road), for any number of cameras with known calibration.
- **Lift:** each pixel is lifted into a frustum of points along its ray, one per discrete depth
  bin, weighted by a predicted depth distribution. Where my pipeline gets depth exactly from
  LiDAR, LSS has to *predict* it, and hedges across depths when unsure.
- **Splat:** the lifted features from all six cameras are sum-pooled into a shared BEV grid of
  pillars: the learned counterpart of my raster's binning step.
- **Shoot:** planning by scoring a set of template trajectories against a cost map predicted in
  BEV. Relevant later, in Phase 3's trajectory evaluation.
- **Training:** end to end from BEV labels only. Depth is never supervised directly; the depth
  distribution is whatever makes the BEV output right.

## My notes

- **Lift and splat aren't networks, and the CNN never sees the calibration.** The only learned
  part is the per-image CNN that predicts a depth distribution α and a feature c per pixel from
  the image alone. Calibration enters in one fixed geometric step,
  p<sub>ego</sub> = R · K⁻¹ · [u·d, v·d, d]ᵀ + t, after undoing the image resize and crop. That
  is my Phase 2 projection run backwards: pixel plus a depth guess to ego, instead of ego to
  pixel.
- **Splat sums everything in a cell instead of choosing one value.** My raster keeps one value
  per cell (nearest, first camera, last drawn). LSS sums every feature that lands in a cell,
  from different depth bins, pixels and cameras, with no z-buffer. Placement is soft, because
  each pixel contributes to every bin along its ray, weighted by α. My own reasoning, not the
  paper's: near cells collect more rays than far ones.
- **Splat borrows the pillar from PointPillars.** A pillar is a grid cell in x and y with no
  binning in z.
    - Every point in the footprint belongs to the same cell regardless of height.
    - z survives as a point feature, encoded by a PointNet and a max over the pillar. That
      avoids 3D convolutions, at the cost of squeezing vertical structure into one vector.
    - It is my raster idea with learned channels.
- **LSS resembles NeRF but runs the other way.** Both discretize rays and weight the samples;
  the table below compares the rest.

| | NeRF | LSS |
|---|---|---|
| The network maps | A 3D position to colour and density | An image to a per-pixel depth distribution |
| Occlusion | Handled with transmittance | None: a softmax with no occlusion term |
| Trained | Fitted per scene | Feed-forward across scenes |
| Supervised by | Pixels | BEV labels |
