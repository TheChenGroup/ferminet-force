from ferminet.base_config import default
from ferminet.train import train
from ferminet.utils import system
from ferminet_force import calc_force
from ferminet_force.estimators import PsiMinZVZBEstimator
from jax import numpy as jnp

cfg = default()
cfg.lock()
cfg.system.molecule = [
    system.Atom(symbol="H", coords=(0, 0, 0)),
    system.Atom(symbol="H", coords=(1.401, 0, 0)),
]
cfg.log.save_path = cfg.log.restore_path = "data"
# It's just tesing, make the network simpler
cfg.network.detnet.determinants = 2
cfg.network.detnet.hidden_dims = ((16, 2), (16, 2))
cfg.system.electrons = (1, 1)
cfg.optim.iterations = 1
cfg.pretrain.iterations = 10
cfg.batch_size = 16

train(cfg)

result = calc_force(cfg, PsiMinZVZBEstimator, steps=2, mcmc_steps=1, mcmc_burn_in=1)

jnp.savez("data/force", **result)
force_all = result["force"]
print("Force mean:")
print(force_all.mean(axis=0))
print("Force stderr:")
print(force_all.std(axis=0))
print("All others:")
for k, v in result.items():
    if isinstance(v, jnp.ndarray):
        print(k)
        print(v.mean(axis=0))
