# Crossing behavior reward configuration

Edit `configs/crossing_rewards.yaml`. The loader and CPU/GPU tensor calculator
are in `scripts/crossing_rewards.py`. This is separate from locomotion rewards:
no checkpoint, 69D actor contract, preview controls or Isaac Lab core is changed.

## Event rewards (once per episode, not per step)

| Event | Reward | Rule |
|---|---:|---|
| Reach goal | +1 | H1 root within 0.6 m of goal centre in XY; no simultaneous failure |
| Safe goal end | +1 | Successful goal only, with no fall or vehicle collision |
| Natural timeout | -1 | Once if neither success nor failure already terminates this transition |
| Vehicle collision | -2 | Actual H1-vehicle contact event; terminate |
| Fall | 0 extra | Terminate, withhold both positive bonuses; no extra penalty requested |

Reaching the goal safely gives +2 in terminal event rewards, in addition to signed
progress and stationary costs. Timeout gives -1 and NO safe bonus.
Manual stop/reset/closing the app is not a timeout penalty event.
Collision or fall overrides simultaneous goal/timeout; a safe goal overrides timeout.
If collision and fall coincide, the default event total remains -2. Progress and
stationary costs are additional. Terminal state is latched: no new reward or cost
until explicit reset. The safe_episode tag is retained for log compatibility,
but now means safe goal completion only.

This goal predicate checks both X and Y. Unlike the manual preview's automatic
stop at Y=4.6, crossing far to the side of the goal pad is not success here.

## Signed goal progress (added 2026-09-25)

For root XY goal distance d before and after each transition:

`progress = progress_reward_per_m * (d_before - d_after)`

Default coefficient is 0.5 reward/m. Moving 1m closer gives +0.5; moving 1m farther
gives -0.5. Standing close to the goal gives zero progress. No extra dt factor.
Previous root XY is initialized at reset, updated only while active, and reset
only for selected environments. Both distances use the same fixed episode goal.
On collision/fall, positive progress in that low-level transition is suppressed;
negative progress remains. Earlier safe substeps can still have earned progress.

The undiscounted sum for a safe round trip is zero; this is not a guarantee against
all discounted-return or stationary-timer exploits. The existing stationary rule
is unchanged. No new fall penalty, action penalty, or training curriculum is added.
From the standard 9.2m start distance to the 0.6m goal radius, progress contributes
about 4.3 points along a successful monotonic path, plus +2 terminal events.

## Increasing stationary penalty

The stationary timer starts at reset and resets when root XY moves at least
`stationary_radius_m` (0.10 m) from its last anchor. This permits accumulated slow
movement but ignores small root jitter and turning in place. No wall-clock time,
camera position or joint vibration is used.

For stationary duration `t` in simulated seconds, the positive penalty rate is:

`rate(t) = min(0.25, 0.05 * max(t - 2.0, 0))` reward units/second.

The per-transition reward is the NEGATIVE integral of this rate over its elapsed
simulation interval, including exact crossings of the grace and cap boundaries.
Do not multiply by dt again. The first 2 seconds are free; the rate is 0.05/s at
3 seconds, 0.15/s at 5 seconds and capped at 0.25/s from 7 seconds onward.
Standing still for 10 seconds costs 1.375 points in total, not 0.25 per step.
The cap applies to the rate, not accumulated episode cost.

The grace, slope, cap and movement radius are YAML parameters. These are initial
design choices, not values validated by training. A grace period allows brief
waiting but is not traffic-aware: waiting safely for a long traffic gap still
incurs a cost. Do not increase this pressure blindly to force unsafe crossing.
Moving in circles or oscillating beyond 0.10 m can reset this timer; it measures
inactivity, NOT progress toward the goal. The separate signed progress term now
rewards net approach and penalizes retreat, but does not change this timer rule.
With a zero fall penalty, a future policy could also exploit falling to escape
future inactivity/timeout costs; monitor that failure mode before accepting a trained model.

## Integration boundary

The calculator is connected to `scripts/behavior_environment.py` and the separate
PPO trainer; see [TRAINING_BEHAVIOR.md](TRAINING_BEHAVIOR.md). The manual play loop
is unchanged and continues using native locomotion `env.step`. The behavior wrapper
steps physics explicitly, detects filtered H1-car contacts every 5 ms, calculates
terminal rewards BEFORE resetting, and sums rewards into each 100 ms transition.
Out-of-bounds ends the episode without a safe bonus or an extra penalty.

Calculator usage (after adding the scripts import path):

```python
from crossing_rewards import CrossingRewardTracker, load_reward_config

rewards = CrossingRewardTracker(num_envs, load_reward_config(), device=device)
rewards.reset(h1_position_xy)

# After executing a behavior action, BEFORE the simulator auto-resets any state:
terms = rewards.step(
    next_h1_position_xy, goal_position_xy,
    fallen=fallen_during_transition,
    vehicle_contact=h1_vehicle_contact_during_transition,
    timeout=natural_episode_timeout,
    dt=elapsed_simulation_seconds,
)
reward = terms["total"]
terminated, truncated = terms["terminated"], terms["truncated"]

# After resetting those environments in physics:
rewards.reset(reset_h1_position_xy, env_ids=done_env_ids)
```

Position tensors are `(num_envs,2)` floating tensors, including each env_origin;
flags are bool `(num_envs,)`, all on the same device. `reset` takes the full position
batch and modifies only selected ids. Log individual returned reward terms to
understand what the behavior policy optimizes. Save the resolved YAML with each run.

Vehicle contact must come from actual robot-vehicle contact filtering, not TTC=0,
distance alone, ground contacts or the preview's generic base-contact termination.
Accumulate collision/fall flags across all low-level steps belonging to one behavior
transition. Use the final pre-reset state and elapsed simulation time, not a
reset pose substituted by an auto-reset wrapper. A transient collision must not be lost.

These terms already include coefficients and time integration. Isaac Lab's native
RewardManager multiplies terms by dt; do not plug this total into it unchanged or
the once-only +1/-2 event rewards and stationary costs would be scaled incorrectly.
Return it directly at the behavior transition boundary, or implement an explicit
adapter with matching semantics and tests.

## Original calculator-only verification (2026-09-24)

This is historical evidence for the earlier reward semantics, not the current
timeout/progress behavior. Current checks are described below.

52 CPU tests passed, including 17 reward tests: YAML defaults/validation, one-shot
goal/safe/collision events, failure precedence, XY goal tolerance, stationary grace,
ramp/cap, invariance to 10/50 Hz timestep for unchanged position, jitter, slow motion,
batch/partial reset and invalid inputs. No simulator or training was run for this
configuration-only change; physical collision detection is not validated by these tests.

```bat
cd /d D:\IssacSim\H1Crossing
D:\IssacSim\env_isaaclab\Scripts\python.exe -B -m unittest discover -s tests -v
```

## Updated reward checks and checkpoint compatibility (2026-09-25)

68 CPU tests pass, including signed approach/retreat, no proximity farming,
one-shot timeout without safe bonus, event precedence, progress terminal latching,
partial reset, translated environments and timestep subdivision invariance.
Run the bounded physics-only check (no PPO updates):

Physical check passed with 4 environments and zero PPO updates:
`logs/20260925T123821_546761Z_reward_smoke/physical_smoke.json`.

```bat
D:\IssacSim\env_isaaclab\Scripts\python.exe -B tests\smoke_crossing_rewards.py
```

The behavior checkpoint contract now includes a reward semantics version. Old
checkpoints keep their original reward contract and cannot be resumed under the
new rewards. Start a NEW behavior run (omit --resume); the frozen locomotion bundle
remains the existing checkpoint500. Existing logs/checkpoints are not overwritten.
