[← All results](results.html)
{: .meta }

# Projection accuracy

{{dates: scripts/build_correctness_report.py scripts/report/build_correctness_report.py}} · [build log §3](build.html#projection) · [checks](validation.html#projection)
{: .meta }

How closely this project's projection of 3D boxes into the cameras reproduces nuScenes'
reference 2D boxes: every one of {{stat:projection.iou_count|,}} annotation and camera pairs
reaches an IoU of {{stat:projection.iou_min|.4f}} or higher. The charts below break that down
by box size, category and camera.
{: .lede }
