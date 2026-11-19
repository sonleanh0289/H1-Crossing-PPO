"""Optional streaming diagnostic, independent of simulation and evaluation gates."""
import csv


class YawTrace:
    FIELDS = ("scenario", "seed", "trial", "time_s", "phase", "target_yaw_rad_s",
              "command_yaw_rad_s", "actual_yaw_rad_s", "signed_error_rad_s", "alive")

    def __init__(self, path, enabled=False):
        self.path, self.enabled = path, enabled
        self.stream = None

    def __enter__(self):
        if self.enabled:
            self.stream = self.path.open("w", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
            self.writer.writeheader()
        return self

    def record(self, scenario, seed, step, dt, target, command, actual, alive):
        if not self.enabled or scenario not in ("turn_left", "turn_right"):
            return
        # CUDA synchronization only in explicitly requested turn diagnostics.
        commands, actuals, survivors = command.tolist(), actual.tolist(), alive.tolist()
        for trial, (cmd, measured, survived) in enumerate(zip(commands, actuals, survivors)):
            self.writer.writerow(dict(scenario=scenario, seed=seed, trial=trial,
                time_s=round((step + 1) * dt, 8), phase="measure" if step >= 250 else "ramp",
                target_yaw_rad_s=target, command_yaw_rad_s=cmd, actual_yaw_rad_s=measured,
                signed_error_rad_s=measured - target, alive=bool(survived)))
        if step % 50 == 49:
            self.stream.flush()

    def __exit__(self, *exc):
        if self.stream is not None:
            self.stream.close()
