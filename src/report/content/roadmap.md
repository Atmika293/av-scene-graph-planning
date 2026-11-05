<!--
DRAFT (Claude): condensed from SpatialVLA_roadmap_updated.md. Deliberately
leaves out the plan's target companies, hardware budget and job-posting
analysis -- add any of that back only if you want it public.
-->
# Roadmap

A seven-phase core sequence, where each phase depends on the one before it, followed by
optional extensions chosen by direction. Phases 1–3 form a complete autonomous-driving project
on their own.
{: .lede }

## Core sequence

| Phase | What it builds | Status |
|---|---|---|
| **1 · Foundations** | SE(3) math, camera calibration and projective geometry | <span class="badge done">Complete</span> |
| **2 · AV ingestion + fusion** | Ray-parallel ingestion, sensor calibration, multi-view projection, geometric BEV fusion, timestamp-sync validation | <span class="badge progress">In progress</span> |
| **3 · Scene graph + VLA ablation** | Open-vocabulary segmentation (Grounding DINO + SAM), 3D instance clustering, CLIP embeddings, scene-graph construction, multi-object tracking; a measured comparison of scene-graph context vs. raw images for VLM trajectory planning (open-loop L2, latency, token cost) | <span class="badge todo">Planned</span> |
| **4 · ROS2 + Nav2 fundamentals** | Nodes, topics and TF2; the Nav2 stack; `robot_localization` EKF | <span class="badge todo">Planned</span> |
| **5 · SLAM + SfM literacy** | Epipolar geometry, pose-graph SLAM, bundle adjustment, structure-from-motion | <span class="badge todo">Planned</span> |
| **6 · Home scene graph + Nav2** | Real indoor capture, pseudo-ground-truth via label projection, quantitative evaluation, language command → Nav2 goal | <span class="badge todo">Planned</span> |
| **7 · Data flywheel** | Hard-example curation, uncertainty triggers, teacher-student distillation, LoRA fine-tuning, evaluation gating | <span class="badge todo">Planned</span> |

## Phase 2 in detail

- <span class="badge done">Done</span> Schema exploration and camera timestamp sync
- <span class="badge done">Done</span> Ingestion module and Parquet tables, validated against the devkit
- <span class="badge done">Done</span> Global → ego → camera → pixel projection, validated
- <span class="badge done">Done</span> Ray-parallelized ingestion and worker-count benchmark
- <span class="badge done">Done</span> Geometric BEV fusion: built, validated, and [published](bev.html)
- <span class="badge todo">Up next</span> Full-dataset ingestion run
- <span class="badge progress">Reading</span> BEV networks: Lift-Splat-Shoot, BEVFormer, BEVFusion, and a survey
- <span class="badge todo">Optional</span> CAN-bus EKF pose estimate vs. nuScenes' ground-truth pose

## Extensions

Optional, independent of each other, and chosen by direction rather than built in sequence.

| Extension | What it adds |
|---|---|
| **A · Person-following** | Reactive closed-loop replanning against a moving target |
| **B · Manipulation policy** | Behaviour cloning (ACT or Diffusion Policy), closed-loop rollout evaluation in simulation |
| **C · Semantic occupancy** | 3D scene completion, and learned camera/LiDAR fusion with a sensor-degradation experiment |
| **D · Closed-loop RL fine-tuning** | RL fine-tuning of the driving policy in a closed-loop simulator |
| **E · RL fundamentals** | From-scratch PPO/SAC on continuous control |
| **F · Synthetic data** | Procedural scene generation, domain randomization, synthetic-to-real transfer |
