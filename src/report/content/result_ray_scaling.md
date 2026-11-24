[← All results](results.html)
{: .meta }

# Ray scaling benchmark: v1.0-mini

Nov 1, 2026 · [build log §4](build.html#ray) · [related experiment](experiment-ray-task-layout.html) · [v1.0-trainval benchmark](result-ray-scaling-trainval.html)
{: .meta }

How ingestion's wall time changes with the number of Ray workers on `v1.0-mini`, run on a
laptop. The best speedup is {{stat:benchmark.best_speedup|.2f}}× at
{{stat:benchmark.best_workers}} workers; the charts below break the time into phases to show
why it stops there.
{: .lede }
