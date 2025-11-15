import numpy as np
import matplotlib.pyplot as plt
from chamber_geometry import *

# Test parameters
pc_design = 2.068e6
thrust_design = 6000
force_coeffcient = 1.4

# Calculate geometry
area_throat = area_throat_calc(pc_design, thrust_design, force_coeffcient)
area_exit = area_exit_calc(diameter_exit_default)
volume_chamber = chamber_volume_calc(area_throat)
area_chamber = area_chamber_calc()
contraction_ratio = contraction_ratio_calc(area_chamber, area_throat)
length_cylindrical = chamber_length_calc(volume_chamber, area_throat, contraction_ratio, theta_default)

r_t = np.sqrt(area_throat / np.pi)
r_c = np.sqrt(area_chamber / np.pi)

print(f"Throat radius: {r_t:.6f} m")
print(f"Chamber radius: {r_c:.6f} m")
print(f"Contraction ratio: {contraction_ratio:.4f}")
print(f"Cylindrical length: {length_cylindrical:.6f} m")
print(f"Chamber volume: {volume_chamber:.6f} m³")

# Get entrance arc start
entrance_arc_start_x, entrance_arc_start_y, tangent_slope = get_nozzle_entrance_arc_start(area_throat)
print(f"\nEntrance arc start: ({entrance_arc_start_x:.6f}, {entrance_arc_start_y:.6f})")
print(f"Tangent slope: {tangent_slope:.6f}")

# Check the 45° line connection
# The 45° line should connect from entrance arc start to cylindrical section
# At cylindrical section: y = r_c
# 45° line: y = y_start + x_start - x (slope = -1)
# At connection: r_c = y_start + x_start - x_cyl_start
x_cyl_start = entrance_arc_start_x + entrance_arc_start_y - r_c
print(f"\nCylindrical section start x: {x_cyl_start:.6f} m")
print(f"Cylindrical section end x: {x_cyl_start - length_cylindrical:.6f} m")

# Verify volume calculation
# Volume = cylindrical volume + conical frustum volume
# But wait, we're using a 45° line, not a full cone
# Let's check what the actual volume would be

# Cylindrical volume
V_cyl = np.pi * r_c**2 * length_cylindrical
print(f"\nCylindrical volume: {V_cyl:.6f} m³")

# The 45° line section volume
# This is a truncated cone from r_c to r_entrance
r_entrance = entrance_arc_start_y
# The 45° line goes from (x_cyl_start, r_c) to (entrance_arc_start_x, r_entrance)
L_line = np.sqrt((x_cyl_start - entrance_arc_start_x)**2 + (r_c - r_entrance)**2)
print(f"45° line length: {L_line:.6f} m")
print(f"45° line horizontal length: {x_cyl_start - entrance_arc_start_x:.6f} m")
print(f"45° line vertical change: {r_c - r_entrance:.6f} m")

# Volume of truncated cone (45° line section)
# For a truncated cone: V = (π/3) * h * (R² + R*r + r²)
# where h is the horizontal length, R is r_c, r is r_entrance
h_line = x_cyl_start - entrance_arc_start_x
V_line = (np.pi / 3) * h_line * (r_c**2 + r_c * r_entrance + r_entrance**2)
print(f"45° line section volume: {V_line:.6f} m³")

# Total volume
V_total = V_cyl + V_line
print(f"\nTotal calculated volume: {V_total:.6f} m³")
print(f"Required volume: {volume_chamber:.6f} m³")
print(f"Difference: {abs(V_total - volume_chamber):.6f} m³")
print(f"Error: {abs(V_total - volume_chamber) / volume_chamber * 100:.2f}%")

