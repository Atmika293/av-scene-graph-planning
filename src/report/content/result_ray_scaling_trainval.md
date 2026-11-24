[← All results](results.html)
{: .meta }

# Ray scaling benchmark: v1.0-trainval

<!-- Hand-dated: see the date note on this result's entry in results.py. -->
Nov 23, 2026 · [build log §8](build.html#full-dataset) · [v1.0-mini benchmark](result-ray-scaling.html)
{: .meta }

How ingestion's wall time changes with the number of Ray workers on the full trainval split
({{stat:benchmark_trainval.scenes}} scenes), next to v1.0-mini
({{stat:benchmark_trainval.mini_scenes}} scenes) on the same
{{stat:benchmark_trainval.cpus}}-CPU machine. Trainval's best is
{{stat:benchmark_trainval.best_speedup|.2f}}× faster than one worker, at
{{stat:benchmark_trainval.best_workers}} workers; mini's is
{{stat:benchmark_trainval.mini_best_speedup|.2f}}×, at
{{stat:benchmark_trainval.mini_best_workers}}. The export from the sweep's last run
{{stat:benchmark_trainval.export_check}}.
{: .lede }
