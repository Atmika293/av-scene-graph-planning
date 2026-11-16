"""Validate bev.multisweep.match_camera_frames_to_sweeps's nearest-in-time camera matching.

For every LIDAR_TOP sweep in every scene (keyframes and the sweeps between
them, not just one sweep window), and every camera, checks the frame
match_camera_frames_to_sweeps selected against a brute-force search
over an independent list of that camera's frames:

  - the independent list comes from scanning nusc.sample_data directly
    (filtered by channel, and by scene via each record's sample_token), not
    from walking the prev/next chain the code under test walks -- so a walk
    that skips frames, repeats them, or strays into another scene can't
    also hide itself here;
  - the brute-force nearest frame is the minimum |t_camera - t_lidar| over
    every one of those frames, which is slow but obviously correct.

A match passes when the selected frame is from the right camera and scene
and its time gap equals the brute-force minimum gap (gaps, not tokens, are
compared, so two equally close frames don't count as a mismatch). A sweep
with no frame within the matcher's max gap must be left unmatched, and a
sweep with one must not be. The max gap is imported from
bev.multisweep.MAX_TIME_GAP_US, the constant the matcher itself passes,
rather than copied here, so the two can't drift apart.

A self-test then feeds the same comparison a deliberately wrong selection
(the frame next to the correct one) and confirms it's reported as a
mismatch, so a check that always passes can't be mistaken for one that
works.

Running this script as __main__ writes the report to
reports/camera_matching_check.txt (overwritten each run): per-camera
tallies, a self-test section, gap statistics, and any mismatch details.
Nothing is printed to stdout.

Run with: python scripts/validation/check_camera_matching.py (from the repo root).
"""

import os
import statistics

from nuscenes.nuscenes import NuScenes

from av_data_pipeline.config import NUSCENES_VERSION, NUSCENES_DATAROOT
from av_data_pipeline.ingestion_keys import KNOWN_SENSOR_CHANNELS, SENSOR_MODALITY_KEYS
from bev.multisweep import MAX_TIME_GAP_US, match_camera_frames_to_sweeps

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "camera_matching_check.txt")
LIDAR_CHANNEL = KNOWN_SENSOR_CHANNELS[SENSOR_MODALITY_KEYS['lidar']][0]
CAMERA_CHANNELS = KNOWN_SENSOR_CHANNELS[SENSOR_MODALITY_KEYS['camera']]
MAX_MISMATCHES_REPORTED = 20


def frames_by_scene_and_channel(nusc: NuScenes) -> dict[tuple[str, str], list[dict]]:
    """Every sample_data record, grouped by (scene token, channel), sorted by timestamp.

    Built from a direct scan of nusc.sample_data -- independent of the
    prev/next chain the code under test walks. Every sample_data record,
    keyframe or not, carries a sample_token, which gives its scene.

    Args:
        nusc: NuScenes devkit instance used to look up records.

    Returns:
        (scene token, channel) -> that channel's records in that scene, in time order.
    """
    scene_by_sample = {sample['token']: sample['scene_token'] for sample in nusc.sample}
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in nusc.sample_data:
        key = (scene_by_sample[record['sample_token']], record['channel'])
        grouped.setdefault(key, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda record: record['timestamp'])
    return grouped


def check_match(lidar_ts: int, selected: dict | None, candidates: list[dict],
                scene_token: str, channel: str, scene_by_sample: dict[str, str]) -> str | None:
    """Compare one selected camera frame against the brute-force nearest frame.

    Args:
        lidar_ts: The lidar sweep's timestamp, in microseconds.
        selected: The camera sample_data record the code under test chose,
            or None if it left this sweep unmatched for this camera.
        candidates: Every frame of this camera in this scene (independent scan).
        scene_token: The scene the lidar sweep belongs to.
        channel: The camera channel being checked.
        scene_by_sample: Sample token -> scene token, to check the selected
            frame's scene.

    Returns:
        None if the selection is correct, otherwise a one-line description of
        what's wrong.
    """
    best_gap = min(abs(frame['timestamp'] - lidar_ts) for frame in candidates)
    should_match = best_gap <= MAX_TIME_GAP_US

    if selected is None:
        return (f"left unmatched, but a frame is {best_gap / 1000:.1f} ms away "
                f"(max gap {MAX_TIME_GAP_US / 1000:.1f} ms)") if should_match else None
    if not should_match:
        return (f"matched, but the nearest frame is {best_gap / 1000:.1f} ms away "
                f"(over the {MAX_TIME_GAP_US / 1000:.1f} ms max gap)")
    if selected['channel'] != channel:
        return f"selected a {selected['channel']} frame for {channel}"
    if scene_by_sample[selected['sample_token']] != scene_token:
        return "selected a frame from a different scene"

    selected_gap = abs(selected['timestamp'] - lidar_ts)
    if selected_gap != best_gap:
        return (f"selected frame is {selected_gap / 1000:.1f} ms away, "
                f"but the nearest is {best_gap / 1000:.1f} ms away")
    return None


if __name__ == "__main__":
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)
    frames = frames_by_scene_and_channel(nusc)
    scene_by_sample = {sample['token']: sample['scene_token'] for sample in nusc.sample}

    verified = {channel: 0 for channel in CAMERA_CHANNELS}
    total = {channel: 0 for channel in CAMERA_CHANNELS}
    unmatched = {channel: 0 for channel in CAMERA_CHANNELS}
    matched_gaps_ms: list[float] = []
    mismatches: list[str] = []
    # (lidar_ts, scene token, channel, wrong frame, that camera's frames), for the self-test
    self_test: tuple[int, str, str, dict, list[dict]] | None = None

    for scene in nusc.scene:
        lidar_sweeps = frames[(scene['token'], LIDAR_CHANNEL)]
        lidar_tokens = [sweep['token'] for sweep in lidar_sweeps]
        lidar_timestamps = [sweep['timestamp'] for sweep in lidar_sweeps]
        # The same anchor assemble_window uses: the scene's last sample's frame for each camera.
        last_sample = nusc.get('sample', scene['last_sample_token'])

        for channel in CAMERA_CHANNELS:
            candidates = frames[(scene['token'], channel)]
            lidar_camera_map = match_camera_frames_to_sweeps(
                nusc, last_sample['data'][channel], lidar_tokens, lidar_timestamps)

            for lidar_token, lidar_ts in zip(lidar_tokens, lidar_timestamps):
                selected_token = lidar_camera_map.get(lidar_token)
                selected = nusc.get('sample_data', selected_token) if selected_token else None
                problem = check_match(lidar_ts, selected, candidates, scene['token'], channel, scene_by_sample)

                total[channel] += 1
                if problem is None:
                    verified[channel] += 1
                    if selected is None:
                        unmatched[channel] += 1
                    else:
                        matched_gaps_ms.append(abs(selected['timestamp'] - lidar_ts) / 1000)
                else:
                    mismatches.append(f"  MISMATCH ({scene['name']}, {channel}, lidar {lidar_token}): {problem}")

                # Self-test material: a correctly matched sweep, plus the frame
                # next to its correct one (strictly farther away in time).
                if self_test is None and problem is None and selected is not None:
                    index = next(i for i, frame in enumerate(candidates) if frame['token'] == selected['token'])
                    neighbour = candidates[index + 1] if index + 1 < len(candidates) else candidates[index - 1]
                    if abs(neighbour['timestamp'] - lidar_ts) != abs(selected['timestamp'] - lidar_ts):
                        self_test = (lidar_ts, scene['token'], channel, neighbour, candidates)

    report = ["--- Nearest-frame check: match_camera_frames_to_sweeps vs. brute force over a table scan ---"]
    for channel in CAMERA_CHANNELS:
        report.append(f"{channel}: {verified[channel]}/{total[channel]}")
    report.append(f"Max gap (bev.multisweep.MAX_TIME_GAP_US): {MAX_TIME_GAP_US / 1000:.1f} ms")

    report.append("")
    report.append("--- Self-test: a deliberately wrong selection must be reported ---")
    if self_test is None:
        report.append("Wrong selection detected: 0/1 (no correctly matched sweep to build the test from)")
    else:
        lidar_ts, scene_token, channel, wrong_frame, candidates = self_test
        caught = check_match(lidar_ts, wrong_frame, candidates, scene_token, channel, scene_by_sample) is not None
        report.append(f"Wrong selection detected: {int(caught)}/1")

    report.append("")
    report.append("--- Gap statistics (diagnostic) ---")
    if matched_gaps_ms:
        report.append(f"Matched gap median (ms) = {statistics.median(matched_gaps_ms):.1f}")
        report.append(f"Matched gap max (ms) = {max(matched_gaps_ms):.1f}")
    for channel in CAMERA_CHANNELS:
        report.append(f"{channel} sweeps left unmatched (no frame within max gap) = {unmatched[channel]}")

    if mismatches:
        report.append("")
        report.append(f"--- Mismatches ({len(mismatches)}, first {MAX_MISMATCHES_REPORTED} shown) ---")
        report.extend(mismatches[:MAX_MISMATCHES_REPORTED])

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w") as outfile:
        outfile.write("\n".join(report) + "\n")
