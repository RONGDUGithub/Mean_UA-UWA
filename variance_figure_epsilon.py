import numpy as np
import matplotlib.pyplot as plt

t = 0.5
epsilon = np.arange(0.9, 1, 0.01)  # numpy.arange does not include the stop value by default

eexp = np.exp(epsilon)
eexp2 = np.exp(epsilon / 2)
exp3 = np.exp(-epsilon / 2)
epsilonstar = 0.61

alpha = np.where(epsilon > epsilonstar, 1 - exp3, 0)

budget = eexp
b = (epsilon * budget - budget + 1) / (2 * budget * (budget - 1 - epsilon))

# high_area  S_h
p = budget / (2 * b * budget + 1)
# low_area  S_l
q = 1 / (2 * b * budget + 1)

Var_sr = np.power((eexp + 1) / (eexp - 1), 2) - np.power(t, 2)
Var_pm = np.power(t, 2) / (eexp2 - 1) + (eexp2 + 3) / (3 * np.power((eexp2 - 1), 2))
Var_hm = alpha * Var_pm + (1 - alpha) * Var_sr

Var_laplace = 8 / np.power(epsilon, 2)
t2 = (t + 1) / 2
Var_sw = 4 * (q * ((1 + 3 * b + 3 * np.power(b, 2) - 6 * b * np.power(t2, 2)) / 3) +
              p * ((6 * b * np.power(t2, 2) + 2 * np.power(b, 3)) / 3) -
              np.power(t2 * 2 * b * (p - q) + q * (b + 1 / 2), 2))

Var_sw_unbiase = Var_sw / (np.power(2 * b * (p - q), 2))

plt.figure()

plt.plot(epsilon, Var_sr, linewidth=2, color='blue')
plt.plot(epsilon, Var_pm, '--', linewidth=2, color='red')
plt.plot(epsilon, Var_hm, ':', linewidth=2, color='green')
plt.plot(epsilon, Var_laplace, '-.', linewidth=2, color='magenta')
plt.plot(epsilon, Var_sw, linewidth=2, color='cyan')
plt.plot(epsilon, Var_sw_unbiase, '--', linewidth=2, color='black')

plt.legend(['SR', 'PM', 'HM', 'Laplace', 'SW', 'SW_{unbiase}'])
plt.show()