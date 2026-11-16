# Experiments

Some decisions in the [build log](build.html) were made on reasoning, not measurement. Each
experiment here goes back to one of them, states the claim as it was originally made, tests
it under a fixed protocol, and reports what the numbers say, whether or not they agree.
{: .lede }

<div class="cards">
<a class="card" href="#ray-task-layout"><strong>Did consolidating Ray tasks matter? <span class="badge diagnostic">Negligible</span></strong><span>Two tasks per batch instead of four: about {{stat:task_layout.diff_pct_max|.0%}} faster after startup at best. Nearly all task time is in bbox projection, which neither layout touches.</span></a>
</div>
