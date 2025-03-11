import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Define robot parameters
# --------------------
a1 = 2
a2 = 4.5
a3 = 7.5

# Z0 is often a fixed vertical offset for joint #2
# Adjust as needed (must satisfy a1^2 - Z0^2 >= 0)
Z0 = 3.75  

# --------------------
# Create arrays of angles (in degrees), then convert to radians
# --------------------
t1_vals_deg = np.linspace(-90, 90, 180)  # Fewer steps for readability
t2_vals_deg = np.linspace(0, 180, 180)
t3_vals_deg = np.linspace(-150, 30, 180)

t1_vals = np.deg2rad(t1_vals_deg)
t2_vals = np.deg2rad(t2_vals_deg)
t3_vals = np.deg2rad(t3_vals_deg)

# --------------------
# Prepare a list (or array) to store (θ1, θ2, θ3, X, Y, Z)
# --------------------
solutions = []

# Precompute the portion used for X0 and Y0
r = np.sqrt(a1**2 - Z0)  # sqrt(a1^2 - Z0^2) must be real and non-negative

# --------------------
# Enumerate all angle combinations
# --------------------
for t1_idx, t1 in enumerate(t1_vals):
    # Coordinates of joint #2
    X0 = r * np.cos(t1)
    Y0 = r * np.sin(t1)
    
    for t2_idx, t2 in enumerate(t2_vals):
        # Intermediate terms for end-effector offset in X-Y plane
        offset_xy = a2 * np.cos(t2) + a3 * np.cos(t2 + t3_vals)
        
        # Compute the corresponding Z for each t3 in a vectorized way:
        Z_array = Z0 + a2 * np.sin(t2) + a3 * np.sin(t2 + t3_vals)
        
        # Compute final X, Y
        X_array = X0 + offset_xy * np.cos(t1)
        Y_array = Y0 + offset_xy * np.sin(t1)
        
        # Store each (θ1, θ2, θ3, X, Y, Z)
        for t3_idx, t3 in enumerate(t3_vals):
            solutions.append([t1_vals_deg[t1_idx], t2_vals_deg[t2_idx], t3_vals_deg[t3_idx], X_array[t3_idx], Y_array[t3_idx], Z_array[t3_idx]])

# --------------------
# Convert to NumPy array for easy plotting
# --------------------
solutions = np.array(solutions)

# --------------------
# Plot Z layers with angle information
# --------------------
unique_z = np.unique(np.round(solutions[:, 5], 1))  # Unique Z values rounded to 1 decimal

print("Available Z layers:", unique_z)

while True:
    z_input = input("Enter Z value to view (from the list above) or 'q' to quit: ")
    
    if z_input.lower() == 'q':
        break
        
    try:
        z_layer = float(z_input)
        
        # Find points close to selected Z layer (within 0.1 units)
        mask = np.abs(solutions[:, 5] - z_layer) < 0.1
        layer_solutions = solutions[mask]

        plt.figure(figsize=(8, 6))
        plt.scatter(layer_solutions[:, 3], layer_solutions[:, 4], s=20, c='blue', label="End Effector Positions")
        
        # Annotate a few points with their corresponding angles
        for i in range(0, len(layer_solutions), max(1, len(layer_solutions)//10)):  # Sample a few points
            x, y = layer_solutions[i, 3], layer_solutions[i, 4]
            t1, t2, t3 = layer_solutions[i, 0], layer_solutions[i, 1], layer_solutions[i, 2]
            plt.annotate(f"({t1:.1f}, {t2:.1f}, {t3:.1f})", (x, y), fontsize=8, color="red")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'XY Points at Z = {z_layer}')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    except ValueError:
        print("Invalid input. Please enter a valid number or 'q' to quit.")
