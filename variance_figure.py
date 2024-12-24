import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-1, 1.01, 0.01)

epsilon = 2

eexp = np.exp(epsilon)
eexp2 = np.exp(epsilon / 2)
exp3 = np.exp(-epsilon / 2)
epsilonstar = 0.61

if epsilon > epsilonstar:
    alpha = 1 - exp3
else:
    alpha = 0

budget = eexp
b = (epsilon * budget - budget + 1) / (2 * budget * (budget - 1 - epsilon))
# high_area  S_h
p = budget / (2 * b * budget + 1)
# low_area  S_l
q = 1 / (2 * b * budget + 1)

Var_sr = ((eexp + 1) / (eexp - 1)) ** 2 - t ** 2
Var_pm = t ** 2 / (eexp2 - 1) + (eexp2 + 3) / (3 * (eexp2 - 1) ** 2)
Var_hm = alpha * Var_pm + (1 - alpha) * Var_sr

Var_laplace = np.ones_like(t) * (8 / epsilon ** 2)
t2 = (t + 1) / 2
Var_sw = 4 * (q * ((1 + 3 * b + 3 * (b ** 2) - 6 * b *( t2 ** 2)) / 3) + p * ((6 * b * (t2 ** 2) + 2 * (b ** 3)) / 3) - (t2 * 2 * b * (p - q) + q * (b + 1/2)) ** 2)
Var_sw_unbiase = Var_sw / (np.ones_like(Var_sw) * (2 * b * (p - q)) ** 2)

plt.figure()

plt.plot(t, Var_sr, linewidth=2, color='blue')
plt.plot(t, Var_pm, '--', linewidth=2, color='red')
plt.plot(t, Var_hm, ':', linewidth=2, color='green')
plt.plot(t, Var_laplace, '-.', linewidth=2, color='magenta')
plt.plot(t, Var_sw, linewidth=2, color='cyan')
plt.plot(t, Var_sw_unbiase, '--', linewidth=2, color='black')

#plt.legend(['SR', 'PM', 'HM', 'Laplace', 'SW', 'SW_unbiase'])
plt.legend(['SR', 'PM', 'HM', 'Laplace', 'SW', 'SW_unbiase'])
plt.title(f'Variance Comparison (epsilon = {epsilon})')

plt.show()