import numpy as np

def inverse_kinematics(X, Y, Z, a1, a2, a3, Z0):
    """
    Compute inverse kinematics solutions that satisfy angle constraints:
    t1: [-90°, 90°]
    t2: [0°, 180°] 
    t3: [-150°, 30°]

    Parameters:
    X, Y, Z : float  -> End-effector target coordinates
    a1, a2, a3 : float  -> Link lengths
    Z0       : float  -> Base height

    Returns:
    Valid (t1, t2, t3) angles in radians that satisfy constraints
    """
    
    # Compute X0 and Y0 using the base radius
    A = np.sqrt(a1**2 - Z0)  # Intermediate step to simplify calculations
    t1 = np.arctan2(Y, X)  # Base rotation
    
    # Check t1 constraint (-90° to 90°)
    if not (-np.pi/2 <= t1 <= np.pi/2):
        raise ValueError("t1 angle outside valid range [-90°, 90°]")
        
    X0 = A * np.cos(t1)  # Base joint X position
    Y0 = A * np.sin(t1)  # Base joint Y position

    # Compute intermediate value D
    D = ((X - X0) ** 2 + (Y - Y0) ** 2 + (Z - Z0) ** 2 - a2**2 - a3**2) / (2 * a2 * a3)

    # Ensure D is within valid range for acos
    if np.abs(D) > 1:
        raise ValueError("Target position is unreachable. Check input values.")

    # Compute both possible t3 (elbow-up and elbow-down)
    t3_up = np.arctan2(np.sqrt(1 - D**2), D)
    t3_down = np.arctan2(-np.sqrt(1 - D**2), D)

    # Function to compute t2 for a given t3
    def compute_t2(t3):
        phi1 = np.arctan2(Z - Z0, np.sqrt((X - X0) ** 2 + (Y - Y0) ** 2))
        phi2 = np.arctan2(a3 * np.sin(t3), a2 + a3 * np.cos(t3))
        return phi1 - phi2

    # Compute t2 for both configurations
    t2_up = compute_t2(t3_up)
    t2_down = compute_t2(t3_down)

    # Check angle constraints and return valid solution
    solutions = []
    
    # Check elbow-up solution
    if (0 <= t2_up <= np.pi and  # t2: [0°, 180°]
        -5*np.pi/6 <= t3_up <= np.pi/6):  # t3: [-150°, 30°]
        solutions.append((t1, t2_up, t3_up))
        
    # Check elbow-down solution
    if (0 <= t2_down <= np.pi and  # t2: [0°, 180°]
        -5*np.pi/6 <= t3_down <= np.pi/6):  # t3: [-150°, 30°]
        solutions.append((t1, t2_down, t3_down))

    if not solutions:
        raise ValueError("No solutions found within angle constraints")
        
    return solutions[0]  # Return first valid solution

# ------------------------------
# Example input values (Target end-effector position)
# ------------------------------
X_target, Y_target, Z_target = 8, 0, 5  # Replace with desired XYZ coordinates
a1, a2, a3 = 2, 4.75, 7.5  # Link lengths
Z0 = 3.75  # Base height

# Compute inverse kinematics
try:
    solution = inverse_kinematics(X_target, Y_target, Z_target, a1, a2, a3, Z0)

    # Print results in degrees for readability
    print("Valid Solution:")
    print(f"t1 = {np.degrees(solution[0]):.2f}°")
    print(f"t2 = {np.degrees(solution[1]):.2f}°")
    print(f"t3 = {np.degrees(solution[2]):.2f}°")

except ValueError as e:
    print(e)
