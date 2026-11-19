# H1 Crossing — stability_v2 checkpoint 500 locomotion

New entrypoint: `scripts/play_crossing_finetuned.py`. Legacy preview/published-policy
scripts are unchanged. No dependency installation or Isaac Lab core edits.
For the separate high-level crossing PPO trainer, see [TRAINING_BEHAVIOR.md](TRAINING_BEHAVIOR.md).
Uses `env_isaaclab` and the adjacent `H1FlatFineTune/h1_finetune` package for the exact
observation/action contract, command smoothing, keyboard and Windows joystick.

The default exported bundle is `checkpoints/h1_flat_stability_v2_0500`, from
`H1FlatFineTune/logs/stability_v2_from_v1_1000_run01/E_1000/checkpoint_0500.pt`.
Its full Stage E evaluation is `H1FlatFineTune/reports/stability_v2_0500_eval`.
Checkpoint SHA-256: `934ad4522e25f2f65f49c9a293059788395c7723c86cd5c2c840d1a7e9654f0d`.
Startup checks
policy/checkpoint/report SHA-256, complete passing stage E evaluation, contract and
command limits. The frozen TorchScript actor has 69 inputs and 19 actions.
Validated on plane at tested speed milestones: forward up to 3 m/s, reverse -1 m/s.
This is NOT a vehicle-avoidance policy or crossing-safety validation.

The previous `checkpoints/h1_flat_stage_e_4000` bundle is preserved unchanged.
To switch back for comparison, add `--bundle checkpoints\h1_flat_stage_e_4000`.
The model switch does not change road geometry, camera mount, input mapping or actuator settings.
Stage E passes the locomotion gate; this is not a separate camera-stability certification.

## CMD — keyboard GUI

```bat
REM Run from H1Crossing with the Isaac Lab Python environment active.
python -B scripts\play_crossing_finetuned.py --keyboard
```

Click the viewport and release all control keys first. W/S: +1/-0.5 m/s; A/D: left/right
yaw; Shift+W/S: +3/-1 m/s; release sends zero command. Space latches stop, Enter resumes
after key release, R resets H1 and car. Focus loss sends zero command. Commands are
acceleration limited, not instantaneous stops. Use keys on the viewport, not text fields.
`--keyboard_speed` and `--keyboard_reverse_speed` adjust the ordinary speeds.

H1 starts at (0,-4.6,1.2), yaw +90 degrees toward +Y. Goal is at Y=+4.6. Car travels +X
at a uniformly sampled 30-50 km/h (8.33-13.89 m/s), with independent uniform spawn
X in [-50,-15] m. Lane Y=-1.75 and vehicle height/orientation remain unchanged.
The camera displays negative Y to the left and positive Y to the right.
R and automatic episode resets restore H1 to the start side and sample a new car pass. Manual play does
not latch stop at the goal by default, allowing retreat; use `--stop_at_goal` if desired.
The automatic goal condition is the Y=4.6 line, not navigation to the exact goal-pad center.

## Flat sidewalks versus legacy curbs

By default, gray sidewalk surfaces are flush with the road and visual-only; the common
plane is the collision surface. Visual road/sidewalk meshes have a 2 mm offset to avoid
z-fighting with the ground plane; this does not change physical height or collisions.
This avoids asking a flat-trained policy to climb the
legacy 15 cm curb. The goal marker is lowered accordingly; road widths/layout unchanged.
`--raised_sidewalks` restores physical 15 cm curbs for an explicit, unvalidated experiment.
The original `preview_crossing_scene.py` and `play_crossing_policy.py` retain old geometry.

## H1 first-person camera (GUI only)

```bat
REM Run from H1Crossing with the Isaac Lab Python environment active.
python -B scripts\play_crossing_finetuned.py --keyboard --h1_camera
```

`--h1_camera` opens an additional **H1 Camera** viewport, leaving the overview intact.
It also works with `--joystick` or `--auto`. Without this flag no camera is created.
Click either simulator viewport, release control keys, then use the same WASD/Shift,
Space/Enter and R controls. You can move or dock the small viewport window in Kit.

The virtual mount follows the physics pose of `torso_link`, offset `(0.20, 0, 0.60)`
metres in its local frame: forward, centred, above the torso origin. It looks along
the robot's forward +X direction (world +Y at spawn); roll/pitch/yaw follow the body.
There is no independent head turn or image stabilization. The camera has no mass
or collision and cannot change robot dynamics.

Configuration is in `scripts/h1_camera.py`, `H1CameraConfig`: 640x480, 90-degree
horizontal FOV, clipping 0.05-100 m. This is a display-only USD camera, **not an
Isaac Lab camera sensor/observation**: no RGB tensors, depth, detection, avoidance,
recording stream or policy retraining. The 69-input/19-action policy stays unchanged.
`--headless --h1_camera` is rejected before simulator startup.
The flag automatically selects Isaac Lab's camera-rendering experience and Vulkan;
you do not need to add `--enable_cameras`. The standard non-camera experience on
this installation produced sky-only images with two viewports, despite valid poses.
No Isaac Lab core or global config file is edited by this implementation.
Rendering is pumped once after each control step, after updating the camera pose.

Use `--steps 300 --capture_viewport` for a bounded capture: the run log contains
`viewport.png` (overview) and `h1_camera.png` (robot), independently of UI focus.
`play_result.json` includes resolved camera config, last mount/camera pose, update
count, control-loop throughput and total GPU-memory snapshots. Throughput is not
monitor FPS; GPU snapshots include other processes and are not peak VRAM.

Manual smoke commands:

```bat
python -B scripts\play_crossing_finetuned.py --auto --walk_speed 0 --h1_camera --steps 300 --capture_viewport
python -B tests\smoke_h1_camera.py
```

The second command uses scripted input samples through the real keyboard mapper
(no OS key injection), exercising forward/reverse/turn/stop/resume/reset for 750
steps with the car parked, with camera-transform assertions at each update.
It also verifies both viewport bindings, the 640x480 image and visible green goal
pixels to catch a regression to sky-only images. Results are saved in
`camera_smoke_assertions.json` before simulator shutdown.

Verified camera-mode standing smoke: 300 steps, finite observations/actions, zero
base-contact falls; both images inspected. Evidence:
`logs/20260918T182207_811881Z_crossing_finetuned`.
On this RTX 4060 8 GB run, total GPU-memory snapshots were 4912-5413 MiB and the
control loop averaged 8.72 steps per wall-clock second. Physics/policy timesteps
remain 0.005/0.02 s, but this dual-view run was slower than real time. Memory and
speed vary with other applications and renderer warmup; these are not peak bounds.

Verified motion/reset smoke: 750 steps, finite values, zero base-contact falls,
one requested reset plus initial reset, and 753 camera-pose checks passed.
Both final images were visually inspected after reset. Evidence:
`logs/20260918T182705_421865Z_crossing_finetuned/camera_smoke_assertions.json`.
The CPU suite now passes all 13 tests, including bounded capture retry/failure
handling. Capture completion is accepted only when a non-empty image file exists.

## Joystick or automatic play

```bat
python -B scripts\play_crossing_finetuned.py --joystick
python -B scripts\play_crossing_finetuned.py --auto --walk_speed 1 --vehicle_speed 0
```

Joystick uses the same Windows mapper as fine-tune: neutral to arm; button 1 resume,
2 stop, 3 reset. Default up/down = forward/reverse, left/right = yaw. Override with
`--joystick_id`, `--forward_axis`, `--turn_axis`, `--forward_sign`, `--turn_sign`.
Only one input mode can be selected. Auto sends constant forward velocity and stops
after crossing the goal line; it does not steer to the pad or react to vehicles.

Normal play samples a new speed and spawn at startup, manual/episode reset and when
the car exceeds X=25 m. By default (`--traffic_mode variable`), each car also
samples a new target speed in 30–50 km/h every 1–3 seconds while moving. It ramps
toward the target at up to 2 m/s² acceleration or 3 m/s² deceleration, without a
position jump. Parameters live in `scripts/vehicle_motion.py:TrafficConfig`.
Use `--traffic_mode constant --yield_probability 0` for the old constant-speed-per-pass behavior.
By default `--yield_probability 0.5` selects a yielding driver independently on
each spawn/reset. That driver brakes for H1 waiting near or occupying the crossing
lane, may stop at zero, and accelerates again after the region clears for 0.5s.
The remaining drivers do not yield. This is not an exact quota or a per-frame coin toss.
Use `--yield_probability 0` to disable or `1` to enable for every driver.
Normal braking is 3m/s², with up to 8m/s² for late yielding. Physical stopping
distance can be insufficient; this is not guaranteed collision avoidance.
No lane changes are made. Respawn is still a teleport, not physical braking.
The default straight road, sidewalks and markings are extended to 110 m to cover
X=-50 m, without moving the crossing, H1, goal or cameras. The vehicle remains kinematic.

`--vehicle_seed 42` reproduces traffic for the same mode, timestep and reset sequence.
Omitting it uses a fresh independent seed, recorded with each pass in
`vehicle_traffic.json` and the final `play_result.json`. Each respawn prints X and km/h.
This RNG does not change the H1 environment seed.

An explicit `--vehicle_speed N` is a legacy diagnostic override in **m/s**, disabling
random speed, yielding AND spawn: start/reset X=-20, wrap X=-25, road length 60 m.
`--vehicle_speed 0` therefore still parks the car at X=-20. Omit this option for random traffic.
Physics: 200 Hz; policy: 50 Hz; H1 command slew: 2 m/s^2 and 1 rad/s^2; moving above 1.5 m/s
limits yaw to 0.25 rad/s, otherwise 0.5 rad/s. No lateral velocity command.

## State-based observations (no camera input)

Behavior reward settings are now available separately in
`configs/crossing_rewards.yaml`; see [REWARDS.md](REWARDS.md) for event timing,
increasing stationary penalties and integration requirements. These rewards are
connected to the separate behavior trainer, not to manual preview; they do not
replace native locomotion rewards in manual play.

The separate `crossing` observation group now contains 8 raw values: relative
vehicle XY, relative goal XY, vehicle world velocity XY, footprint TTC and TTC
validity. The frozen actor still consumes ONLY the unchanged 69D `policy` group.
See [OBSERVATIONS.md](OBSERVATIONS.md) for the ordered schema, coordinate frame,
TTC assumptions, reset lifecycle, recording and the boundary before behavior training.
Use `--record_crossing_observations` to save a 50 Hz CSV and print values every
100 steps; no camera tensors are added, even with the display camera enabled.

## Verification and artifacts

### Crossing observations verification (2026-09-22)

- 35 CPU tests passed: TTC footprint/time intervals/validity, vectorized coordinate
  translation, scripted velocity after teleport, group isolation and existing regressions.
- Headless 650-step standing smoke with random traffic and a wrap: finite values,
  zero falls; all 651 recorded rows checked against traffic event positions and speeds.
  Evidence: `logs/20260922T015428_273879Z_crossing_finetuned`.
- Headless lifecycle smoke: 250 steps, two automatic episode resets, 251 rows;
  post-reset relative vehicle/goal positions and new speeds asserted before shutdown.
  Evidence: `logs/20260922T015710_703253Z_crossing_finetuned/observation_smoke_assertions.json`.
- Frozen checkpoint, camera and traffic-controller source hashes unchanged; native
  locomotion contract passed at runtime; Isaac Lab core Git status clean.
  No behavior training or GUI test was run for this observation-only change.

### Random traffic verification (2026-09-21)

- 23 CPU tests pass, covering speed units/bounds, independent RNG replay, constant
  speed between passes, reset/wrap sampling, fixed-speed compatibility and camera tests.
- Headless smoke: 550 steps, seed 42, H1 standing; finite values and zero base-contact falls.
  Initial pass: X=-27.62 m at 30.50 km/h. After X>25: X=-40.37 m at 34.46 km/h.
  Resolved road length is 110 m. Evidence:
  `logs/20260921T165011_531549Z_crossing_finetuned/play_result.json`.
- Checkpoint bundles and camera/input code are unchanged; Isaac Lab core remains clean.
  No training or GUI interaction test was run for this change. This smoke is not
  a collision-avoidance or crossing-safety evaluation.

### Stability v2 checkpoint 500 integration (2026-09-21)

- Export and copied-bundle checksums match the passing checkpoint-500 evaluation.
- 14 CPU tests pass, including default-bundle selection, legacy override, contract,
  hash rejection, exported actor equivalence and unchanged camera geometry tests.
- Headless default-bundle smoke: 650 steps, 1 m/s command, vehicle parked at X=-20;
  finite observations/actions, zero base-contact falls, goal line reached.
  Start: `(0, -4.6, 1.2)`; final: approximately `(1.68, 5.08, 0.92)`.
  This is goal-line crossing, not exact navigation to the pad centre.
  Evidence: `logs/20260921T151029_059978Z_crossing_finetuned/play_result.json`.
- Legacy bundle, geometry/camera/input scripts and fine-tune package source checksums
  are unchanged; Isaac Lab core Git status remains clean. No training was run.
- GUI camera and physical keyboard/joystick interaction were not re-run for this
  model switch. To view the new model with S commanding -1 m/s:

```bat
REM Run from H1Crossing with the Isaac Lab Python environment active.
python -B scripts\play_crossing_finetuned.py --keyboard --keyboard_reverse_speed 1.0 --h1_camera
```

```bat
python -B -m unittest discover -s tests -v
python -B scripts\play_crossing_finetuned.py --headless --auto --walk_speed 0 --steps 300
```

Each run creates its own `logs/<timestamp>_crossing_finetuned` with provenance,
resolved environment config and `play_result.json`, or `failure.json` on failure.
`--capture_viewport` (GUI only) saves a viewport PNG when the run ends.
`falls` currently counts native base-contact termination, not a complete vehicle
collision metric. Falling on a car or manual driving errors are not prevented.
Collision-specific terminations, crossing rewards and behavior learning are now
available in the separate behavior trainer. Manual play does not use them;
stage E's pass applies to flat locomotion only, not autonomous crossing.

### Historical E-4000 integration checks

- 7 CPU integration tests passed (bundle integrity, actor equivalence, geometry,
  argument validation and source compilation).
- Headless, 300 steps, zero command with moving car: finite values, no base-contact falls.
  Evidence: `logs/20260917T180819_045983Z_crossing_finetuned/play_result.json`.
- Headless, 650 steps, 1 m/s with car parked: crossed the Y=4.6 goal line, no falls.
  Final position approximately (1.77, 5.14, 0.93): lateral drift remains; this is not
  a navigation controller that reaches the exact goal pad.
  Evidence: `logs/20260917T180959_181286Z_crossing_finetuned/play_result.json`.
- GUI, 100 steps, zero command with moving car: finite values, no falls. Viewport
  capture inspected: upright H1 and car, start on the left, goal opposite, no road
  z-fighting after the visual-only offset fix.
  Evidence: `logs/20260917T182235_042686Z_crossing_finetuned/viewport.png` and
  the adjacent `play_result.json`.
- Existing crossing scripts and fine-tune package source hashes unchanged; Isaac Lab
  core Git status clean. No training or package installation performed.

Physical keyboard/gamepad interaction was not re-tested during these automated
crossing checks; both reuse the fine-tune input mappers without changing their code.
