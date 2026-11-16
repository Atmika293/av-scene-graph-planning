# Bird's-eye-view fusion

LiDAR points in the ego frame, each coloured by the camera that sees it, with nuScenes'
annotated boxes overlaid, shown as a scatter, a multi-channel raster (colour, height, intensity),
and in 3D. Everything shown here comes from code checked in the [BEV fusion
checks](validation.html#bev); how it was built is in the [build log](build.html).
{: .lede }

## Sweeps, keyframes and coordinate frames

Each sample is a keyframe partway through its scene, so the **Sweeps** menu can also show it
merged with the LiDAR sweeps captured in the half second before it, which fills in far more of
the ground ([multisweep checks](validation.html#multisweep)).

- **Sweep:** one full turn of the roof LiDAR (LIDAR_TOP). It spins at 20 Hz in the
  nuScenes setup, so each sweep is about 50 ms of points. Every sweep is recorded.
- **Keyframe:** a sweep that nuScenes also chose as a *sample*, about every tenth sweep. The
  table below compares the two.
- **Ego frame:** coordinates attached to the car: origin at the rear axle, x forward, y left,
  z up. The car moves between sweeps, so each sweep has its own ego frame, wherever the car
  was at that sweep's timestamp. Every view on this page is drawn in the **keyframe's ego
  frame**, the one at the keyframe's timestamp.
- **Global frame:** coordinates fixed to the map. nuScenes records the car's pose (position
  and heading) in it for every sweep and camera image, which is what lets points move between
  ego frames taken at different times.

| | Keyframe | Other sweeps |
|---|---|---|
| How often | 2 Hz, about every half second | The sweeps in between (20 Hz overall) |
| Annotated boxes | Yes: objects around the car are labelled with 3D boxes | None |
| Linked to other sensors | One sample record groups it with each camera's and radar's keyframe | Stand alone; flagged as not a keyframe |
| On this page | The sample shown, and the source of every box | Added by **Multisweep** |

## How the BEV is built

### One sweep

The "1 (keyframe)" setting.

1. **Read the sweep.** Each point comes with x, y, z and intensity, in the LiDAR's own frame.
2. **LiDAR to ego to global.** The LiDAR's mounting position and rotation on the car move the
   points into the ego frame. The car's pose at the sweep's timestamp then moves them into the
   global frame.
3. **Global to each camera.** For each camera, the points go from the global frame into that
   camera's ego frame, using the car's pose at the *image's* timestamp, then into the camera's
   own frame using its mounting. Points behind the camera are dropped.
4. **Camera to pixels.** The camera's intrinsic matrix projects each remaining point onto the
   image: pixel = K · (x, y, z) / z, where z is depth. Points that land outside the image are
   dropped, and the rest take the image's colour at that pixel, interpolated between
   neighbouring pixels.
5. **Pick a colour.** The cameras are tried in a fixed order and each point keeps the first
   colour it gets. A point no camera sees stays uncoloured (blue on the charts). There is no
   occlusion test, so a point hidden from a camera by something nearer takes that nearer
   object's colour.
6. **Top-down view.** The coloured points, in the keyframe's ego frame, are plotted by x and y
   (scatter), or binned into a square grid (raster). In each cell the raster keeps the point
   nearest the sensor and records its colour, height (z) and intensity. Points tied for nearest
   are averaged.

### Multisweep

1. **Gather the window.** The keyframe's sweep plus the sweeps just before it. For each sweep,
   every camera's image nearest in time is picked, if one is within the maximum time gap;
   otherwise that camera is skipped for that sweep.
2. **Colour each sweep on its own.** Steps 1 to 5 above run once per sweep, with that sweep's
   own car pose and its own matched images, so each sweep is coloured by images from nearly the
   same moment.
3. **Global to the keyframe's ego frame.** Each sweep's points, already in the global frame, are
   moved into the keyframe's ego frame with the inverse of the car's pose at the keyframe.
   Anything that stood still lines up across sweeps; anything that moved leaves a trail.
4. **One raster per sweep.** Each sweep is binned on the same grid as above. "Nearest the
   sensor" is measured from that sweep's own ego origin, where the LiDAR actually was when it
   took the point, not from the keyframe's.
5. **Merge, newest first.** Each cell takes every channel (colour, height, intensity) from the
   newest sweep with a point in it. A cell no sweep reached stays empty.

Pick a view, a sample and how many sweeps below. Each sample's data loads when you select it.
