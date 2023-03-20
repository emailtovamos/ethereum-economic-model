import math
import matplotlib.pyplot as plt
import numpy as np

def non_linear_increase(x: float) -> float:
    # Set the maximum value of the output to 0.7
    max_value = 0.7

    # Set the minimum value of the output to 0.04
    min_value = 0.04

    # Set the rate of increase as a function of x
    rate_of_increase = (max_value - min_value) / (1 + math.exp(-0.5 * x))
    rate_of_increase = max_value - (max_value - min_value) / (1 + x**2)

    # Calculate the current value of the output based on the rate of increase
    current_value = min_value + rate_of_increase

    # Return the current value
    return current_value

samples = [non_linear_increase(x) for x in range(100)]
samples = np.array(samples)

mean = np.mean(samples, axis=0)
std = np.std(samples, axis=0)

plt.plot(samples)
# plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5)
plt.show()