# Validation

Every stage of the pipeline is checked against a reference that's independent of the code
under test, before the next stage is allowed to build on it. A bug then surfaces at the stage
that introduced it, rather than as a vague wrongness at the end of the pipeline.
{: .lede }

The table below is regenerated from each check's own latest output every time this site is
rebuilt; none of its numbers are typed by hand. Each row links to the script that produced it.

<div class="callout" markdown="1">
**What "independent" means here.** A check that reuses the code it's checking can only
confirm the code agrees with itself. So each check compares against something else: the
devkit's own loaders and tables, a reference implementation written separately, nuScenes'
annotations, or a physical fact (a car drives forward, not sideways).
</div>
