"""Per-environment post-reset capture barrier, without simulator state access.

Camera.frame counts buffer reads, not physical exposures. Do not use it to
certify freshness. Drain two scheduled render/capture cycles after a reset;
only a later, strictly newer simulation timestamp may enter perception.
This is a bounded pipeline flush, not hardware exposure timestamp validation.
"""
import math

RESET_CAPTURE_POLICY = "drain-two-scheduled-captures-v1"


class CameraResetBarrier:
    def __init__(self, count):
        self.generation = [0] * count
        self.reset_time = [None] * count
        self.last_capture = [None] * count
        self.remaining = [0] * count
        self.discarded = [0] * count
        self.first_accepted = [None] * count
        self.status = ["ready"] * count

    def reset(self, ids, now):
        if not math.isfinite(now):
            raise ValueError("Reset timestamp must be finite")
        for i in ids:
            self.generation[i] += 1
            self.reset_time[i] = now
            self.last_capture[i] = now
            self.remaining[i] = 2
            self.first_accepted[i] = None
            self.status[i] = "reset_wait"

    def admit(self, i, now):
        previous = self.last_capture[i]
        if not math.isfinite(now) or (previous is not None and now <= previous + 1e-9):
            self.discarded[i] += 1
            self.status[i] = "nonadvancing_capture"
            return False
        self.last_capture[i] = now
        if self.remaining[i]:
            self.remaining[i] -= 1
            self.discarded[i] += 1
            self.status[i] = "reset_drain"
            return False
        if self.first_accepted[i] is None:
            self.first_accepted[i] = now
        self.status[i] = "ready"
        return True

    def snapshot(self, i):
        return {"camera_reset_generation": self.generation[i],
                "camera_reset_time_s": self.reset_time[i],
                "camera_capture_barrier_status": self.status[i],
                "camera_reset_frames_remaining": self.remaining[i],
                "camera_first_accepted_capture_s": self.first_accepted[i]}

    def report(self):
        return {"policy": RESET_CAPTURE_POLICY,
                "discarded_captures_by_env": self.discarded.copy(),
                "environments": [self.snapshot(i) for i in range(len(self.generation))],
                "scope": "Scheduled-render drain; no native exposure timestamp assertion"}
