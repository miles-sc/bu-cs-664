# Engine Size membership functions, per your spec
# x in liters [1.5, 6.5]; y in [0, 1]
# Requires: numpy, matplotlib, scikit-fuzzy (pip install scikit-fuzzy)

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Universe of discourse (dense for smooth lines)
x_min, x_max = 1.5, 6.5
x = np.linspace(x_min, x_max, 2001)

# --- Membership functions ---
# very small: 1.0 at 1.5, linearly down to 0.0 at 2.4, then 0.0
very_small = fuzz.trapmf(x, [1.5, 1.5, 1.5, 2.4])  # left-shoulder with zero-length plateau

# small: 0 at 1.5 -> 1 at 2.4 -> 0 at 3.5 (triangular)
small = fuzz.trimf(x, [1.5, 2.4, 3.5])

# medium: 0 at 2.4 -> 1 at 3.5 -> 0 at 5.0 (triangular)
medium = fuzz.trimf(x, [2.4, 3.5, 5.0])

# large: 0 at 3.5 -> 1 at 5.0 -> 0 at 6.5 (triangular)
large = fuzz.trimf(x, [3.5, 5.0, 6.5])

# very large: 0 at 5.0 -> 1 at 6.5 (right-shoulder)
very_large = fuzz.trapmf(x, [5.0, 6.5, 6.5, 6.5])  # right-shoulder with zero-length plateau

# --- Plot ---
plt.figure(figsize=(9, 5))
plt.plot(x, very_small, linewidth=2, label="very small")
plt.plot(x, small, linewidth=2, label="small")
plt.plot(x, medium, linewidth=2, label="medium")
plt.plot(x, large, linewidth=2, label="large")
plt.plot(x, very_large, linewidth=2, label="very large")

plt.xlim(x_min, x_max)
plt.ylim(0.0, 1.0)
plt.xlabel("Engine Size (L)")
plt.ylabel("Membership value")
plt.title("Membership Functions: Engine Size")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
