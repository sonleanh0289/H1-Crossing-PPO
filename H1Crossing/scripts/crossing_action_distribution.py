"""Bounded PPO actions using an affine Beta distribution, a = 2*x - 1.

The environment's physical speed mapping is unchanged. Unlike an unbounded
Gaussian followed by hard clipping, samples and PPO densities share a bounded
support. Shapes >1 keep the density finite at the endpoints. Near-boundary
concentration can still be learned and must be monitored.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta, kl_divergence
from rsl_rl.modules.distribution import Distribution


def concentrations(output):
    alpha, beta = output.chunk(2, dim=-1)
    return F.softplus(alpha) + 1., F.softplus(beta) + 1.


class BetaMean(nn.Module):
    def forward(self, output):
        alpha, beta = concentrations(output)
        return 2.*alpha/(alpha+beta)-1.


class BoundedBetaDistribution(Distribution):
    def __init__(self, output_dim, init_std=.5):
        super().__init__(output_dim)
        if not 0 < init_std < 1/math.sqrt(3):
            raise ValueError("Beta initialization needs 0 < init_std < 1/sqrt(3)")
        self.init_std = init_std
        self._distribution = None

    @property
    def input_dim(self):
        return 2*self.output_dim

    def update(self, mlp_output):
        self._distribution = Beta(*concentrations(mlp_output), validate_args=False)

    def sample(self):
        return 2.*self._distribution.sample()-1.

    def deterministic_output(self, mlp_output):
        return BetaMean()(mlp_output)

    def as_deterministic_output_module(self):
        return BetaMean()

    @property
    def mean(self):
        return 2.*self._distribution.mean-1.

    @property
    def std(self):
        return 2.*self._distribution.variance.sqrt()

    @property
    def entropy(self):
        return (self._distribution.entropy()+math.log(2.)).sum(-1)

    @property
    def params(self):
        return self._distribution.concentration1, self._distribution.concentration0

    def log_prob(self, outputs):
        # Only numerical endpoint protection, not a Gaussian action clipping path.
        x = (outputs+1.)*.5
        eps = torch.finfo(x.dtype).eps
        density = (self._distribution.log_prob(x.clamp(eps,1.-eps))-math.log(2.)).sum(-1)
        outside = (outputs.abs()>1).any(-1)
        return torch.where(outside, -torch.inf, density)

    def kl_divergence(self, old_params, new_params):
        # KL is invariant to the same invertible affine transform on both Betas.
        return kl_divergence(Beta(*old_params), Beta(*new_params)).sum(-1)

    def init_mlp_weights(self, mlp):
        head = [layer for layer in mlp if isinstance(layer, nn.Linear)][-1]
        nn.init.orthogonal_(head.weight, gain=.01)
        shape = (1/self.init_std**2-1)/2
        nn.init.constant_(head.bias, math.log(math.expm1(shape-1)))
