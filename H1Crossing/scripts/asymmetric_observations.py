"""Privileged critic inputs only; no vehicle truth is copied to actor inputs."""
import torch
from crossing_observations import planar_ttc

FORMAT = "h1-crossing-asymmetric-v1"
EXTRA_FIELDS = ("gt_car_dx/50", "gt_car_dy/10", "gt_car_vx/15", "gt_car_vy/15",
                "gt_ttc/10", "gt_ttc_valid")


def critic_observation(sensor, h1_xy, h1_velocity, car_xy, car_velocity):
    """23D sensor prefix + six current GT features, evaluated at decision time.

    Vehicle velocity is instantaneous scripted world XY, never differenced over
    respawn. TTC assumes constant relative velocity, not future driver intent.
    This function has no cache, RNG, sensor mutation, or action side effects.
    """
    if sensor.ndim != 2 or sensor.shape != (len(h1_xy), 23) or not torch.isfinite(sensor).all():
        raise ValueError("Asymmetric actor observations must be finite (N,23)")
    relative = car_xy - h1_xy
    ttc, valid = planar_ttc(relative, car_velocity - h1_velocity)
    gt = torch.cat((relative / relative.new_tensor([50., 10.]), car_velocity / 15.,
                    ttc / 10., valid.to(ttc.dtype)), dim=-1)
    if not torch.isfinite(gt).all():
        raise ValueError("Privileged observations must be finite")
    return torch.cat((sensor, gt), dim=-1)


def critic_contract(actor_fields):
    if len(actor_fields) != 23:
        raise ValueError("Asymmetric critic requires the camera/radar 23D actor schema")
    return {"format": FORMAT, "actor_observation_group": "behavior", "critic_observation_group": "critic",
            "critic_observation_fields": list(actor_fields) + list(EXTRA_FIELDS),
            "privileged_information": "current scripted vehicle world-XY state; constant-velocity TTC",
            "gt_timestamp": "current simulation decision time; sensor exposure age is retained in actor prefix",
            "play_requires_privileged_inputs": False}
