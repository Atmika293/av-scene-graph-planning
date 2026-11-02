"""Validate get_sample_tokens_from_scene against nuScenes' own field2token lookup.

For every scene in the loaded dataset, compares the sample tokens returned by
ingestion.get_sample_tokens_from_scene (walking the sample linked list via each
sample's 'next' field) against nusc.field2token('sample', 'scene_token', ...)
(a direct table scan) — as sets, since traversal order isn't guaranteed to
match a table scan's order and only token membership is being verified here.

Running this script as __main__ writes a structured report — verified count,
and any mismatch details — to reports/verify_sample_tokens.txt (overwritten
each run). Nothing is printed to stdout; every result goes into the report file.
"""

import os

from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion import get_sample_tokens_from_scene


def verify_scene(nusc: NuScenes, scene_token: str) -> tuple[bool, str]:
    """Compare get_sample_tokens_from_scene against nusc.field2token for one scene.

    Args:
        nusc: NuScenes devkit instance used to look up records.
        scene_token: Token of the scene to check.

    Returns:
        A (matched, detail) pair. matched is True if both methods return the
        same set of sample tokens; detail is a one-line summary on success, or
        the set difference on mismatch.
    """
    via_linked_list = set(get_sample_tokens_from_scene(nusc, scene_token))
    via_field2token = set(nusc.field2token('sample', 'scene_token', scene_token))

    if via_linked_list == via_field2token:
        return True, f"{len(via_linked_list)} sample tokens match"

    only_in_linked_list = sorted(via_linked_list - via_field2token)
    only_in_field2token = sorted(via_field2token - via_linked_list)
    return False, (f"only in get_sample_tokens_from_scene: {only_in_linked_list}; "
                   f"only in field2token: {only_in_field2token}")


if __name__ == "__main__":
    REPORT_DIR = "reports"
    REPORT_FILE = os.path.join(REPORT_DIR, "verify_sample_tokens.txt")

    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    verified = 0
    mismatches: list[tuple[str, str, str]] = []  # (scene_token, scene_name, detail)

    for scene in nusc.scene:
        matched, detail = verify_scene(nusc, scene['token'])
        if matched:
            verified += 1
        else:
            mismatches.append((scene['token'], scene['name'], detail))

    report = ["--- get_sample_tokens_from_scene vs nusc.field2token ---",
             f"Verified: {verified}/{len(nusc.scene)}"]
    for scene_token, scene_name, detail in mismatches:
        report.append(f"  MISMATCH ({scene_token}, {scene_name}): {detail}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(report) + "\n")
