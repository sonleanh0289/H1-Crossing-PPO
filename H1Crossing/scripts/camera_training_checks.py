"""Smoke-only proof that camera observations reach the PPO actor and storage."""
import torch


class CameraRolloutAudit:
    def __init__(self, observation_dim=22):
        if observation_dim not in (22, 23):
            raise ValueError("Expected camera 22D or camera+radar 23D")
        self.observation_dim = observation_dim
        self.actor_batches = 0
        self.storage_batches = 0
        self.transitions = 0
        self.position_valid = 0
        self.velocity_valid = 0

    def act(self, alg, obs):
        expected = obs["behavior"].detach().clone()
        if expected.ndim != 2 or expected.shape[1] != self.observation_dim or not torch.isfinite(expected).all():
            raise AssertionError("Sensor PPO input has incorrect shape or nonfinite values")
        calls = []
        def verify_input(module, inputs):
            if not torch.equal(inputs[0]["behavior"], expected):
                raise AssertionError("Actor did not receive the camera observation")
            calls.append(True)
        hook = alg.actor.register_forward_pre_hook(verify_input)
        try:
            action = alg.act(obs)
        finally:
            hook.remove()
        if len(calls) != 1 or not torch.equal(alg.transition.observations["behavior"], expected):
            raise AssertionError("PPO transition does not match the actor's camera input")
        self.actor_batches += 1
        return action, expected

    def check_storage(self, alg, index, expected):
        if not torch.equal(alg.storage.observations["behavior"][index], expected):
            raise AssertionError("Rollout storage changed/replaced the camera observation")
        self.storage_batches += 1
        self.transitions += len(expected)
        self.position_valid += int((expected[:,19] > .5).sum())
        self.velocity_valid += int((expected[:,20] > .5).sum())

    def report(self):
        return {"actor_input_dim":self.observation_dim, "actor_batches_checked":self.actor_batches,
                "storage_batches_checked":self.storage_batches, "transitions_checked":self.transitions,
                "position_valid_transitions":self.position_valid,
                "velocity_valid_transitions":self.velocity_valid,
                "actor_and_storage_match":self.actor_batches == self.storage_batches > 0}
