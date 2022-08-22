# Copyright 2022 Allan Chain
# Part of the code is based on FermiNet by DeepMind, same license.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ferminet.base_config import default
from ferminet.train import train
from ferminet.utils import system
from ferminet_force import calc_force
from ferminet_force.estimators import SWCTEstimator
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

result = calc_force(cfg, SWCTEstimator, steps=2, mcmc_steps=1, mcmc_burn_in=1)

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
