"""
Helper functions to generate stochastic environmental processes
"""

import numpy as np
from stochastic import processes
import stochastic as st

import experiments.simulation_configuration as simulation
from experiments.utils import rng_generator


def create_eth_price_process(
    timesteps=simulation.TIMESTEPS,
    dt=simulation.DELTA_TIME,
    rng=np.random.default_rng(1),
    minimum_eth_price=1500,
):
    """Configure environmental ETH price process

    > A Brownian excursion is a Brownian bridge from (0, 0) to (t, 0) which is conditioned to be non-negative on the interval [0, t].

    See https://stochastic.readthedocs.io/en/latest/continuous.html
    """
    process = processes.continuous.BrownianExcursion(t=(timesteps * dt), rng=rng)
    samples = process.sample(timesteps * dt + 1)
    maximum_eth_price = max(samples)
    samples = [
        minimum_eth_price + eth_price_sample / maximum_eth_price * minimum_eth_price
        for eth_price_sample in samples
    ]
    return samples

def create_linear_increasing_aa_process(
    timesteps=simulation.TIMESTEPS,
    dt=simulation.DELTA_TIME,
):
    # x = st.Arange(0, 1, 0.1).sample()
    x = st.Arange(0, timesteps * dt + 1, dt).sample()

# aa process we are keeping same as validator adoption rate and we will set it to constant, constant*0.5 and constant*2
# for normal, low and high aa adoption just like validator adoption for simplicity
# So just like how many validators are springing up per epoch, here this means how many eths are coming up in this epoch
# due to aa related mev
def aa_process(
    timesteps=simulation.TIMESTEPS, # 360
    dt=simulation.DELTA_TIME, # 225
    rng=np.random.default_rng(1),
    aa_adoption_rate=4,
):

    process = processes.continuous.PoissonProcess(
        rate=1 / aa_adoption_rate, rng=rng
    )
    samples = process.sample(timesteps * dt + 1)
    samples = np.diff(samples)
    samples = [int(sample) for sample in samples]
    return samples

def create_validator_process(
    timesteps=simulation.TIMESTEPS,
    dt=simulation.DELTA_TIME,
    rng=np.random.default_rng(1),
    validator_adoption_rate=4,
):
    """Configure environmental validator staking process

    > A Poisson process with rate lambda is a count of occurrences of i.i.d. exponential random variables with mean 1/lambda. This class generates samples of times for which cumulative exponential random variables occur.

    See https://stochastic.readthedocs.io/en/latest/continuous.html
    """
    process = processes.continuous.PoissonProcess(
        rate=1 / validator_adoption_rate, rng=rng
    )
    samples = process.sample(timesteps * dt + 1)
    samples = np.diff(samples)
    samples = [int(sample) for sample in samples]
    return samples


def create_stochastic_process_realizations(
    process,
    timesteps=simulation.TIMESTEPS,
    dt=simulation.DELTA_TIME,
    runs=5,
):
    """Create stochastic process realizations

    Using the stochastic processes defined in `processes` module, create random number generator (RNG) seeds,
    and use RNG to pre-generate samples for number of simulation timesteps.
    """

    switcher = {
        "eth_price_samples": [
            create_eth_price_process(timesteps=timesteps, dt=dt, rng=rng_generator())
            for _ in range(runs)
        ],
        "validator_samples": [
            create_validator_process(timesteps=timesteps, dt=dt, rng=rng_generator())
            for _ in range(runs)
        ],
        "validator_uptime_samples": [
            rng_generator().uniform(0.96, 0.99, timesteps * dt + 1) for _ in range(runs)
        ],
    }

    return switcher.get(process, "Invalid Process")
