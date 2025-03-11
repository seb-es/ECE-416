import numpy as np

def calculate_positions(t1, t2, t3, a1, a2, a3, Z0):
    """
    Compute X, Y, Z forward kinematics positions based on joint angles and link lengths.

    Parameters:
    t1, t2, t3 : float (radians)  -> Joint angles
    a1, a2, a3 : float            -> Link lengths
    Z0         : float            -> Base height

    Returns:
    X, Y, Z, X0, Y0 : float       -> Calculated positions
    """

    # Compute X0 and Y0
    A = np.sqrt(a1**2 - Z0)  # Intermediate step to simplify calculations
    X0 = A * np.cos(t1)
    Y0 = A * np.sin(t1)

    # Compute X, Y, Z
    X = X0 + (a2 * np.cos(t2) + a3 * np.cos(t2 + t3)) * np.cos(t1)
    Y = Y0 + (a2 * np.cos(t2) + a3 * np.cos(t2 + t3)) * np.sin(t1)
    Z = Z0 + a2 * np.sin(t2) + a3 * np.sin(t2 + t3)

    return X, Y, Z, X0, Y0

# Example input values
t1 = np.radians(0)  # Convert degrees to radians
t2 = np.radians(79.95)
t3 = np.radians(-107.14)
a1, a2, a3 = 2, 4.75, 7.5  # Link lengths
Z0 = 3.75  # Base height

# Compute positions
X, Y, Z, X0, Y0 = calculate_positions(t1, t2, t3, a1, a2, a3, Z0)

# Print results
print(f"X0 = {X0:.2f}, Y0 = {Y0:.2f}")
print(f"X = {X:.2f}, Y = {Y:.2f}, Z = {Z:.2f}")
