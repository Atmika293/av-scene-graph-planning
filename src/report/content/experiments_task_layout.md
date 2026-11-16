[← All experiments](experiments.html)
{: .meta }

# Did consolidating Ray tasks matter?

{{dates: benchmark_task_layout.py}} · revisits [build log §4](build.html#ray) · [build log §7](build.html#experiments) · [data](#data)
{: .meta }

**The decision.** Ingestion first wrote each batch with four Ray tasks, one per output table:
three core tables (samples, sample data, annotations) and the projected 2D bboxes. I merged
the three core-table writes into one task, leaving two tasks per batch.

**The claim being tested.** As the build log puts it, splitting the core writes "only paid
scheduling and deserialization overhead per task without adding real parallel compute." That
predicts the two-task layout should be measurably faster, most of all at low worker counts,
where extra tasks wait in the queue.

**Why it wasn't data-driven.** The reasoning was about Ray task overhead in general. I never
timed the four-task layout under the benchmark protocol before switching, so the build log
states an explanation nothing had measured.

**Protocol.** The four-task layout was reconstructed as a config switch
(`task_layout = "split"` in `config.toml`); every other code path is shared. Both layouts ran
through the same harness as the [main benchmark](result-ray-scaling.html): same subprocess
isolation, batch size, in-flight cap and worker counts ({{stat:task_layout.worker_counts}}),
{{stat:task_layout.repeats}} repeats each. The two layouts' runs were interleaved, with the
order swapped on every repeat, so slow drift (heat, background load) lands on both equally.
The metric is *time after startup*: wall time minus the fixed startup cost (interpreter,
imports, `ray.init()`, loading the devkit's tables), which is the same under both layouts
and would otherwise bury the difference.

**Result.** The claim holds in direction, but the effect is tiny. Two tasks was faster by at
most {{stat:task_layout.diff_pct_max|.1%}}. That gap is consistent at
{{stat:task_layout.faster_workers}} workers and within run to run noise at
{{stat:task_layout.noise_workers}}.

**Diagnosis.** The same runs record each task's compute time, so the time can be split by
task type. The bbox projection task is {{stat:task_layout.bbox_share_min|.0%}} to
{{stat:task_layout.bbox_share_max|.0%}} of all task compute in every configuration. The three
core-table writes together take {{stat:task_layout.core_compute_min|.2f}} to
{{stat:task_layout.core_compute_max|.2f}} seconds per run, so merging their tasks could only
ever save a slice of that. Splitting halves each core task's work but doubles their number,
and the totals barely move.

**What changed.** The code stays at two tasks per batch: no slower, and simpler. The build
log's explanation now links here. The process changed too: a design choice that the build log
justifies with a performance argument gets a measurement before it is written up.

**Limits and scope.**

- One machine: an Apple M3 laptop (8 cores, 16 GB), on `v1.0-mini`.
- {{stat:task_layout.repeats}} repeats per configuration. "Within noise" is a rule of thumb
  (the difference is smaller than the two standard deviations added together), not a
  significance test.
- On the full dataset each task does more work, so fixed per-task overhead should matter even
  less. That is a prediction, not a measurement.
- Bbox compute grows from {{stat:task_layout.bbox_compute_fewest_workers|.1f}} s per run at
  {{stat:task_layout.fewest_workers}} worker to {{stat:task_layout.bbox_compute_most_workers|.1f}} s
  at {{stat:task_layout.most_workers}}, which suggests the workers contend for something
  (memory bandwidth, or cores shared with the driver). This experiment doesn't explain it.

**Open thread.** Since bbox projection is where the time goes, it is the next thing worth
measuring.
