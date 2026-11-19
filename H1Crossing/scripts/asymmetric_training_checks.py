"""Audit real actor/critic routing and the first rollout without logging GT frames."""
import torch


class AsymmetricRolloutAudit:
    def __init__(self):
        self.batches = 0
        self.stored = 0
        self.transitions = 0

    def act(self, alg, obs):
        expected = obs.detach().clone()
        sensor, critic = expected["behavior"], expected["critic"]
        if sensor.shape != (len(sensor),23) or critic.shape != (len(sensor),29):
            raise AssertionError("Asymmetric inputs must be actor23/critic29")
        if not torch.equal(sensor,critic[:,:23]) or not torch.isfinite(critic).all():
            raise AssertionError("Critic prefix must be the exact sensor snapshot and inputs finite")
        # Deterministic actor outputs must not depend on privileged values or key presence.
        changed = expected.clone()
        changed["critic"] = torch.full_like(critic, 1234.)
        output = alg.actor(expected)
        if not torch.equal(output,alg.actor(changed)) or not torch.equal(output,alg.actor(expected.select("behavior"))):
            raise AssertionError("Privileged critic inputs leaked to actor")
        calls = []
        def record_actor(module, inputs):
            if not torch.equal(inputs[0], sensor):
                raise AssertionError("Actor MLP input is not the 23D sensor tensor")
            calls.append("actor")
        def record_critic(module, inputs):
            if not torch.equal(inputs[0], critic):
                raise AssertionError("Critic MLP input is not the 29D privileged tensor")
            calls.append("critic")
        hooks = [alg.actor.mlp.register_forward_pre_hook(record_actor), alg.critic.mlp.register_forward_pre_hook(record_critic)]
        try:
            actions = alg.act(obs)
        finally:
            for hook in hooks:
                hook.remove()
        if calls != ["actor", "critic"]:
            raise AssertionError("Unexpected PPO actor/critic calls")
        self.batches += 1
        return actions, expected

    def check_storage(self, alg, index, expected):
        for name in ("behavior", "critic"):
            if not torch.equal(alg.storage.observations[name][index],expected[name]):
                raise AssertionError("Rollout storage does not match decision-time inputs")
        self.stored += 1
        self.transitions += len(expected)

    def report(self):
        return {"actor_input_dim":23,"critic_input_dim":29,"batches_checked":self.batches,
                "transitions_checked":self.transitions,
                "actor_ignores_privileged_inputs":self.batches>0,
                "actor_critic_storage_match":self.batches==self.stored>0}
