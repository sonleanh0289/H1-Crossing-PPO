"""Pure-Python, independently seeded scripted traffic; no simulator imports."""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class TrafficConfig:
    speed_kmh: tuple[float, float] = (40.0, 60.0)
    spawn_x_m: tuple[float, float] = (-50.0, -15.0)
    exit_x_m: float = 25.0
    mode: str = "variable"
    acceleration_mps2: float = 2.0
    deceleration_mps2: float = 3.0
    target_interval_s: tuple[float, float] = (1.0, 3.0)
    yield_probability: float = 0.5
    yield_stop_x_m: float = -5.5  # vehicle CENTER; front is 2.1m ahead
    yield_crossing_half_width_m: float = 2.5
    yield_wait_y_min_m: float = -5.5
    yield_clearance_m: float = 0.4
    yield_clear_delay_s: float = 0.5
    emergency_deceleration_mps2: float = 8.0
    vehicle_palette: str = "legacy_green"

    def validate(self):
        lo, hi = self.speed_kmh
        xlo, xhi = self.spawn_x_m
        tlo, thi = self.target_interval_s
        values = (lo, hi, xlo, xhi, self.exit_x_m, tlo, thi,
                  self.acceleration_mps2, self.deceleration_mps2, self.yield_probability,
                  self.yield_stop_x_m, self.yield_crossing_half_width_m, self.yield_wait_y_min_m,
                  self.yield_clearance_m, self.yield_clear_delay_s, self.emergency_deceleration_mps2)
        if (not all(math.isfinite(v) for v in values)
                or not 0 < lo <= hi or not xlo <= xhi < 0 < self.exit_x_m
                or not 0 < tlo <= thi or min(self.acceleration_mps2, self.deceleration_mps2) <= 0
                or self.mode not in ("constant", "variable") or not 0 <= self.yield_probability <= 1
                or self.yield_stop_x_m >= 0 or self.yield_crossing_half_width_m <= 0
                or min(self.yield_clearance_m, self.yield_clear_delay_s) < 0
                or self.emergency_deceleration_mps2 < self.deceleration_mps2
                or self.vehicle_palette not in ("legacy_green", "multicolor")):
            raise ValueError("Invalid traffic motion configuration")

    def contract(self):
        self.validate()
        result = {"mode": self.mode, "speed_kmh": list(self.speed_kmh),
                  "spawn_x_m": list(self.spawn_x_m), "exit_x_m": self.exit_x_m}
        if self.mode == "variable":
            result.update(acceleration_mps2=self.acceleration_mps2,
                          deceleration_mps2=self.deceleration_mps2,
                          target_interval_s=list(self.target_interval_s), dynamics_version=1)
        if self.yield_probability > 0:
            result["yielding"] = {"version": 1, "probability_per_pass": self.yield_probability,
                "stop_center_x_m": self.yield_stop_x_m, "crossing_half_width_m": self.yield_crossing_half_width_m,
                "wait_y_min_m": self.yield_wait_y_min_m, "clearance_m": self.yield_clearance_m,
                "clear_delay_s": self.yield_clear_delay_s, "emergency_deceleration_mps2": self.emergency_deceleration_mps2}
        if self.vehicle_palette != "legacy_green":
            result["appearance"] = {"palette": self.vehicle_palette, "version": 1,
                                    "selection": "seed-env-pass-balanced-cycle"}
        return result

    @property
    def road_length_m(self):
        # Centred road plus 5 m clearance for the 4.2 m car at either end.
        return 2 * (max(abs(self.spawn_x_m[0]), abs(self.exit_x_m)) + 5.0)


class VehicleMotion:
    """Uniform target speeds with bounded acceleration during each pass.

    Explicit fixed_speed_mps preserves the legacy diagnostic trajectory,
    including parking at X=-20 when speed is zero.
    """
    def __init__(self, *, seed=None, fixed_speed_mps=None, config=TrafficConfig()):
        config.validate()
        if fixed_speed_mps is not None and (not math.isfinite(fixed_speed_mps) or fixed_speed_mps < 0):
            raise ValueError("Fixed vehicle speed must be finite and nonnegative")
        self.config = config
        self.seed = random.SystemRandom().getrandbits(32) if seed is None else seed
        self.rng = random.Random(self.seed)
        self.fixed_speed_mps = fixed_speed_mps
        self.x = -20.0
        self.speed_mps = 0.0
        self.target_speed_mps = 0.0
        self.target_remaining_s = 0.0
        self.sim_time_s = 0.0
        self.events = []
        self.will_yield = False
        self.yield_clear_remaining_s = 0.0
        self.is_yielding = False

    def reset(self, reason="reset"):
        if self.fixed_speed_mps is None:
            self.x = self.rng.uniform(*self.config.spawn_x_m)
            self.speed_mps = self.rng.uniform(*self.config.speed_kmh) / 3.6
        else:
            self.x = -25.0 if reason == "wrap" else -20.0
            self.speed_mps = self.fixed_speed_mps
        self.target_speed_mps = self.speed_mps
        self.target_remaining_s = 0.0
        if self.fixed_speed_mps is None and self.config.mode == "variable":
            self._resample_target()
        # Keep legacy RNG draws unchanged when yielding is explicitly disabled.
        self.will_yield = (self.fixed_speed_mps is None and self.config.yield_probability > 0
                           and self.rng.random() < self.config.yield_probability)
        self.yield_clear_remaining_s = 0.0
        self.is_yielding = False
        event = {"pass": len(self.events) + 1, "reason": reason,
                 "sim_time_s": self.sim_time_s, "spawn_x_m": self.x,
                 "speed_mps": self.speed_mps, "speed_kmh": self.speed_mps * 3.6}
        self.events.append(event)
        return event

    def _resample_target(self):
        self.target_speed_mps = self.rng.uniform(*self.config.speed_kmh) / 3.6
        self.target_remaining_s = self.rng.uniform(*self.config.target_interval_s)

    def step(self, dt, h1_xy=None, vehicle_y=-1.75):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Traffic timestep must be finite and positive")
        self.sim_time_s += dt
        if self.fixed_speed_mps is None:
            # Resolve target timers at step boundaries (200 Hz in behavior).
            if self.config.mode == "variable" and self.target_remaining_s <= 1e-6:
                self._resample_target()
            cfg = self.config
            request = False
            if h1_xy is not None:
                if len(h1_xy) != 2 or not all(math.isfinite(v) for v in (*h1_xy, vehicle_y)):
                    raise ValueError("H1/vehicle coordinates must be finite local XY")
                request = (abs(h1_xy[0]) <= cfg.yield_crossing_half_width_m
                           and cfg.yield_wait_y_min_m <= h1_xy[1] <= vehicle_y+.9+.35+cfg.yield_clearance_m)
            if self.will_yield and request:
                self.yield_clear_remaining_s = cfg.yield_clear_delay_s
            else:
                self.yield_clear_remaining_s = max(0., self.yield_clear_remaining_s-dt)
            self.is_yielding = (self.will_yield and (request or self.yield_clear_remaining_s > 0)
                               and self.x <= cfg.yield_crossing_half_width_m+2.1+.35)
            target = self.target_speed_mps
            brake = cfg.deceleration_mps2
            if self.is_yielding:
                distance = max(0., cfg.yield_stop_x_m-self.x-.25)
                target = min(target, math.sqrt(2*cfg.deceleration_mps2*distance))
                required = self.speed_mps**2 / (2*max(distance, .01))
                brake = min(cfg.emergency_deceleration_mps2, max(brake, required))
            delta = target - self.speed_mps
            rate = cfg.acceleration_mps2 if delta >= 0 else brake
            ramp_time = min(dt, abs(delta) / rate)
            dv = math.copysign(rate * ramp_time, delta)
            self.x += self.speed_mps * dt + dv * (dt - .5 * ramp_time)
            self.speed_mps += dv
            self.target_remaining_s -= dt
        else:
            self.x += self.speed_mps * dt
        if self.x > self.config.exit_x_m:
            self.reset("wrap")
            return True
        return False

    def report(self):
        return {"mode": "random" if self.fixed_speed_mps is None else "fixed_legacy",
                "seed": self.seed, "speed_range_kmh": self.config.speed_kmh,
                "spawn_x_range_m": self.config.spawn_x_m, "exit_x_m": self.config.exit_x_m,
                "fixed_speed_mps": self.fixed_speed_mps, "x_m": self.x,
                "speed_mps": self.speed_mps, "events": self.events}
