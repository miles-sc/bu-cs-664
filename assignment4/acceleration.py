# 0-60 Acceleration membership functions
# x in seconds [2.7, 9.1]; y in [0, 1]
# Requires: numpy, matplotlib, scikit-fuzzy

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Universe of discourse
x_min, x_max = 2.7, 9.1
x = np.linspace(x_min, x_max, 3001)

# --- Membership functions ---
# very fast: 1.0 at 2.7 -> 0.0 at 5.5
very_fast = fuzz.trapmf(x, [2.7, 2.7, 2.7, 5.5])

# fast: 0.0 at 2.7 -> 1.0 at 5.5 -> 0.0 at 6.2
fast = fuzz.trimf(x, [2.7, 5.5, 6.2])

# medium: 0.0 at 5.5 -> 1.0 at 6.2 -> 0.0 at 8.3
medium = fuzz.trimf(x, [5.5, 6.2, 8.3])

# slow: 0.0 at 6.2 -> 1.0 at 8.3 -> 0.0 at 9.1
slow = fuzz.trimf(x, [6.2, 8.3, 9.1])

# very slow: 0.0 at 8.3 -> 1.0 at 9.1
very_slow = fuzz.trapmf(x, [8.3, 9.1, 9.1, 9.1])

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(x, very_fast, linewidth=2, label="very fast")
plt.plot(x, fast, linewidth=2, label="fast")
plt.plot(x, medium, linewidth=2, label="medium")
plt.plot(x, slow, linewidth=2, label="slow")
plt.plot(x, very_slow, linewidth=2, label="very slow")

plt.xlim(x_min, x_max)
plt.ylim(0.0, 1.0)
plt.xlabel("0–60 Acceleration (s)")
plt.ylabel("Membership value")
plt.title("Membership Functions: 0–60 Acceleration")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
