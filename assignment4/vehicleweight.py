# Vehicle Weight membership functions
# x in lbs [2200, 6200]; y in [0, 1]
# Requires: numpy, matplotlib, scikit-fuzzy (pip install scikit-fuzzy)

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Universe of discourse
x_min, x_max = 2200, 6200
x = np.linspace(x_min, x_max, 4001)

# --- Membership functions ---
# very light: 1.0 at 2200, down to 0.0 at 3200
very_light = fuzz.trapmf(x, [2200, 2200, 2200, 3200])

# light: 0 at 2200 -> 1 at 3200 -> 0 at 3700
light = fuzz.trimf(x, [2200, 3200, 3700])

# medium: 0 at 3200 -> 1 at 3700 -> 0 at 4500
medium = fuzz.trimf(x, [3200, 3700, 4500])

# heavy: 0 at 3700 -> 1 at 4500 -> 0 at 6200
heavy = fuzz.trimf(x, [3700, 4500, 6200])

# very heavy: 0 at 4500 -> 1 at 6200
very_heavy = fuzz.trapmf(x, [4500, 6200, 6200, 6200])

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(x, very_light, linewidth=2, label="very light")
plt.plot(x, light, linewidth=2, label="light")
plt.plot(x, medium, linewidth=2, label="medium")
plt.plot(x, heavy, linewidth=2, label="heavy")
plt.plot(x, very_heavy, linewidth=2, label="very heavy")

plt.xlim(x_min, x_max)
plt.ylim(0.0, 1.0)
plt.xlabel("Vehicle Weight (lbs)")
plt.ylabel("Membership value")
plt.title("Membership Functions: Vehicle Weight")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
