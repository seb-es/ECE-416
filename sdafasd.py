import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import keyboard

# ---------------------------
# Robot and Base Parameters
# ---------------------------
# Base position parameters (used for forward kinematics visualization)
# (In this formulation, the "base" of the moving arm is computed from a link-length offset.)
# The given code uses these values:
X0, Y0, Z0 = 0.5, 0, 3.75  # Base position (Z0 used as vertical offset)
a1, a2, a3 = 2, 4.75, 7.5     # Link lengths

# ---------------------------
# Initial Joint Angles (in radians)
# ---------------------------
t1 = 0.0  # Base rotation angle
t2 = 0.0  # Shoulder angle
t3 = 0.0  # Elbow angle

# Angle step size (in radians)
angle_step = np.deg2rad(2)  # 2 degrees per key press

# ---------------------------
# Create 3D Figure for Visualization
# ---------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.set_zlim([0, 15])
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3-DOF Robotic Arm Simulation (Forward Kinematics)")

# Create a line object for the arm segments
line, = ax.plot([], [], [], 'o-', lw=3, markersize=8, color='blue')
# Create a text box to show joint angles and end-effector coordinates
text_handle = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# ---------------------------
# Forward Kinematics Function
# ---------------------------
def calculate_positions(t1, t2, t3):
    """
    Computes forward kinematics given joint angles t1, t2, t3.
    
    Using the formulas:
      X0_local = A * cos(t1)    where A = sqrt(a1^2 - Z0)
      Y0_local = A * sin(t1)
      
      X1 = X0_local + a2 * cos(t2) * cos(t1)
      Y1 = Y0_local + a2 * cos(t2) * sin(t1)
      Z1 = Z0 + a2 * sin(t2)
      
      X2 = X1 + a3 * cos(t2+t3) * cos(t1)
      Y2 = Y1 + a3 * cos(t2+t3) * sin(t1)
      Z2 = Z1 + a3 * sin(t2+t3)
      
    Returns:
      A tuple of three lists: (x_coords, y_coords, z_coords) for the base, first joint, and end-effector.
      
    Change: The starting point for the first line segment is now (0,0,0)
    instead of (0,0,Z0). The second point remains (X0_local, Y0_local, Z0).
    """
    # Compute base radius (using the provided formulation)
    A = np.sqrt(a1**2 - Z0)
    X0_local = A * np.cos(t1)
    Y0_local = A * np.sin(t1)
    
    # First joint position (after the shoulder link)
    X1 = X0_local + a2 * np.cos(t2) * np.cos(t1)
    Y1 = Y0_local + a2 * np.cos(t2) * np.sin(t1)
    Z1 = Z0 + a2 * np.sin(t2)
    
    # End-effector position (after the elbow link)
    X2 = X1 + a3 * np.cos(t2 + t3) * np.cos(t1)
    Y2 = Y1 + a3 * np.cos(t2 + t3) * np.sin(t1)
    Z2 = Z1 + a3 * np.sin(t2 + t3)
    
    # Updated: The starting point for the first segment is (0,0,0)
    return ([0, X0_local, X1, X2],
            [0, Y0_local, Y1, Y2],
            [0, Z0, Z1, Z2])

# ---------------------------
# Animation Update Function
# ---------------------------
def update(frame):
    global t1, t2, t3

    # Check keyboard inputs to adjust joint angles.
    if keyboard.is_pressed('z'):
        t1 += angle_step
    elif keyboard.is_pressed('x'):
        t1 -= angle_step

    if keyboard.is_pressed('c'):
        t2 += angle_step
    elif keyboard.is_pressed('v'):
        t2 -= angle_step

    if keyboard.is_pressed('b'):
        t3 += angle_step
    elif keyboard.is_pressed('n'):
        t3 -= angle_step

    # Calculate joint positions using forward kinematics
    x_vals, y_vals, z_vals = calculate_positions(t1, t2, t3)

    # Update the arm plot with the new joint positions
    line.set_data(x_vals, y_vals)
    line.set_3d_properties(z_vals)

    # Compute the end-effector position (last point in the lists)
    end_effector_pos = (x_vals[-1], y_vals[-1], z_vals[-1])
    text_handle.set_text(
        "t₁: {:.1f}°, t₂: {:.1f}°, t₃: {:.1f}°\nEnd-effector: ({:.2f}, {:.2f}, {:.2f})".format(
            np.rad2deg(t1), np.rad2deg(t2), np.rad2deg(t3),
            end_effector_pos[0], end_effector_pos[1], end_effector_pos[2]
        )
    )
    return line, text_handle

# ---------------------------
# Run the Animation Loop
# ---------------------------
ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=False)
plt.show()
