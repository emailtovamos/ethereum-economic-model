import matplotlib.pyplot as plt
import numpy as np
from stochastic import processes

def aa_process(
    timesteps=360, # 360
    dt=225, # 225
    rng=np.random.default_rng(1),
    aa_adoption_rate=0.01,
):

    process = processes.continuous.PoissonProcess(
        rate=1 / aa_adoption_rate, rng=rng
    )
    samples = process.sample(timesteps * dt + 1)
    # samples = np.diff(samples)
    # samples = [int(sample) for sample in samples]
    return samples

samples = [aa_process() for _ in range(100)]
samples = np.array(samples)

mean = np.mean(samples, axis=0)
std = np.std(samples, axis=0)

plt.plot(mean)
plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)
plt.show()

