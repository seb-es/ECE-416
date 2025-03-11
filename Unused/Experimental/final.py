import numpy as np  # Scientific computing library

def axis_angle_rot_matrix(k, q):
    """
    Creates a 3x3 rotation matrix in 3D space from an axis and an angle.
    """
    c_theta = np.cos(q)
    s_theta = np.sin(q)
    v_theta = 1 - c_theta
    kx, ky, kz = k[0], k[1], k[2]

    # First row of the rotation matrix
    r00 = kx*kx*v_theta + c_theta
    r01 = kx*ky*v_theta - kz*s_theta
    r02 = kx*kz*v_theta + ky*s_theta

    # Second row of the rotation matrix
    r10 = kx*ky*v_theta + kz*s_theta
    r11 = ky*ky*v_theta + c_theta
    r12 = ky*kz*v_theta - kx*s_theta

    # Third row of the rotation matrix
    r20 = kx*kz*v_theta - ky*s_theta
    r21 = ky*kz*v_theta + kx*s_theta
    r22 = kz*kz*v_theta + c_theta

    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix

def hr_matrix(k, t, q):
    """
    Create the Homogeneous Representation matrix using axis-angle.
    """
    R = axis_angle_rot_matrix(k, q)
    t = np.array(t).reshape(3,1)
    T = np.concatenate((R, t), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    return T

def orientation_error(R_goal, R_current):
    """
    Computes the orientation error as a 3-element vector.
    Here we use the skew-symmetric part of R_err = R_goal * R_current^T.
    """
    R_err = R_goal @ R_current.T
    skew = 0.5 * (R_err - R_err.T)
    err_vec = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    return err_vec

class RoboticArm:
    def __init__(self, k_arm, t_arm):
        """
        k_arm: 2D array listing the axes (one row per joint).
        t_arm: 2D array listing the translation from the previous joint to current.
        """
        self.k = np.array(k_arm)
        self.t = np.array(t_arm)
        assert k_arm.shape == t_arm.shape, 'Warning! Improper definition of rotation axes and translations'
        self.N_joints = k_arm.shape[0]

    def position(self, Q, index=-1, p_i=[0,0,0]):
        """
        Compute the position (in the global frame) of a point given in a joint frame.
        """
        p_i = np.array(p_i).reshape(3,1)
        point = np.concatenate((p_i, np.array([[1]])), axis=0)
        
        if index == -1:
            index = self.N_joints - 1

        T = np.eye(4)
        for i in range(index+1):
            T = T @ hr_matrix(self.k[i], self.t[i], Q[i])
        p_global = T @ point
        return p_global[:3, 0]

    def full_transformation(self, Q, p_eff_N=[0,0,0]):
        """
        Compute the full 4x4 transformation from the base to the end effector.
        p_eff_N is the end-effector offset in the last joint's frame.
        """
        T = np.eye(4)
        for i in range(self.N_joints):
            T = T @ hr_matrix(self.k[i], self.t[i], Q[i])
        p_eff_N = np.array(p_eff_N).reshape(3,1)
        T_eff = np.concatenate((np.eye(3), p_eff_N), axis=1)
        T_eff = np.concatenate((T_eff, np.array([[0, 0, 0, 1]])), axis=0)
        return T @ T_eff

    def full_jacobian(self, Q, p_eff_N=[0,0,0]):
        """
        Computes the 6xN Jacobian for the end effector pose.
        The top 3 rows are for linear velocity; the bottom 3 for angular velocity.
        """
        T = np.eye(4)
        joint_positions = []
        joint_axes = []
        for i in range(self.N_joints):
            T = T @ hr_matrix(self.k[i], self.t[i], Q[i])
            joint_positions.append(T[:3, 3])
            # For a revolute joint, the joint axis (local z) transformed to base frame:
            joint_axes.append(T[:3, :3] @ np.array([0, 0, 1]))
        
        T_eff = self.full_transformation(Q, p_eff_N)
        p_eff = T_eff[:3, 3]
        
        J_linear = []
        J_angular = []
        for i in range(self.N_joints):
            p_i = joint_positions[i]
            z_i = joint_axes[i]
            Jp = np.cross(z_i, (p_eff - p_i))
            J_linear.append(Jp.reshape(3,1))
            J_angular.append(z_i.reshape(3,1))
        
        J_linear = np.hstack(J_linear)   # 3xN
        J_angular = np.hstack(J_angular)   # 3xN
        J_full = np.vstack((J_linear, J_angular))  # 6xN
        return J_full

    def reduced_jacobian(self, Q, p_eff_N=[0,0,0]):
        """
        For a 5 DOF arm, one rotational DOF is missing.
        Here we assume the uncontrolled rotation is about the end-effector's z-axis.
        We remove that row from the angular part of the full Jacobian.
        (i.e. drop the 6th row).
        """
        J_full = self.full_jacobian(Q, p_eff_N)  # 6xN
        J_reduced = np.delete(J_full, 5, axis=0)   # Remove row index 5 => now 5xN
        return J_reduced

    def pseudo_inverse_reduced(self, theta_start, p_eff_N, goal_position, R_goal, max_steps=500):
        """
        Inverse kinematics for a 5 DOF arm (position: 3 DOF, orientation: 2 DOF).
        We use the reduced 5xN Jacobian and a 5D error vector.
        The orientation error is computed in full (3D) and then we drop the
        component corresponding to the uncontrolled rotation.
        """
        v_step_size = 0.05
        theta_max_step = 0.2
        Q = theta_start.copy()
        
        # Compute current pose:
        T_eff = self.full_transformation(Q, p_eff_N)
        p_current = T_eff[:3, 3]
        R_current = T_eff[:3, :3]
        
        # Compute full orientation error (3D)
        delta_theta_full = orientation_error(R_goal, R_current)
        # Remove the component that is not controlled. Here we assume the 3rd component is missing.
        delta_theta = delta_theta_full[:2]  # keep only first 2 components
        
        delta_p = goal_position - p_current  # 3D position error
        
        # Form the 5D error vector: [delta_p; delta_theta]
        error = np.concatenate((delta_p, delta_theta))
        i = 0
        while np.linalg.norm(error) > 0.01 and i < max_steps:
            print(f"Iteration {i}: Q = {Q}, Error norm = {np.linalg.norm(error):.4f}")
            
            # Scale the error (optional gain)
            error_step = error * v_step_size / np.linalg.norm(error)
            
            # Get the reduced 5xN Jacobian
            J_red = self.reduced_jacobian(Q, p_eff_N)
            # Compute the pseudoinverse (J_red is 5xN, here N=5 so square or tall)
            J_inv = np.linalg.pinv(J_red)
            dQ = J_inv @ error_step  # joint updates (N x 1)
            dQ = np.clip(dQ, -theta_max_step, theta_max_step)
            Q = Q + dQ
            
            # Recompute the error:
            T_eff = self.full_transformation(Q, p_eff_N)
            p_current = T_eff[:3, 3]
            R_current = T_eff[:3, :3]
            delta_p = goal_position - p_current
            delta_theta_full = orientation_error(R_goal, R_current)
            delta_theta = delta_theta_full[:2]  # reduced orientation error
            error = np.concatenate((delta_p, delta_theta))
            i += 1
        
        print(f"Converged in {i} iterations with error norm {np.linalg.norm(error):.4f}.")
        return Q

    # (Keep your original position and jacobian methods if needed.)

    def jacobian(self, Q, p_eff_N=[0,0,0]):
        """
        Existing Jacobian for position only (3xN).
        """
        p_eff = self.position(Q, -1, p_eff_N)
        first_iter = True
        jacobian_matrix = None
        for i in range(self.N_joints):
            p_eff_minus_this_p = p_eff - self.position(Q, index=i)
            k = np.array(self.k[i])
            this_jacobian = np.cross(k, p_eff_minus_this_p).reshape(3,1)
            if first_iter:
                jacobian_matrix = this_jacobian
                first_iter = False
            else:
                jacobian_matrix = np.concatenate((jacobian_matrix, this_jacobian), axis=1)
        return jacobian_matrix

def main():
    """
    Example: A 5 DOF arm with the base as a joint.
    The arm has 5 revolute joints, but only 5 DOF so one orientation DOF is missing.
    We design the IK in a reduced (5D) space: 3 for position and 2 for orientation.
    """
    # Define rotation axes for 5 joints (all rotate about the z-axis)
    k = np.array([
        [0, 0, 1],  # Joint 1 (base)
        [0, 0, 1],  # Joint 2
        [0, 0, 1],  # Joint 3
        [0, 0, 1],  # Joint 4
        [0, 0, 1]   # Joint 5
    ])

    # Define translations (example values)
    a1 = 2
    a2 = 4
    a3 = 4
    a4 = 2
    a5 = 2
    # Note: Our base is a joint, so the first translation might be [0,0,0]
    t = np.array([
        [0, 0, 0],           # Base to Joint 1
        [0.25, 0, a1],          # Joint 1 to Joint 2
        [a2, 0, 0],          # Joint 2 to Joint 3
        [0, 0, 0],           # Joint 3 to Joint 4
        [0, 0, a3+a4]        # Joint 4 to Joint 5
    ])

    # End-effector offset in the last joint's frame (example)
    p_eff = [0, -a5, 0]

    # Create the arm object
    arm = RoboticArm(k, t)

    # Starting joint angles (5 DOF)
    q_0 = np.array([0, 0, 0, 0, 0])

    # Desired end-effector position (in base frame)
    goal_position = np.array([8, 2, a1])

    # Desired end-effector orientation (use a rotation matrix)
    # For example, a 45° rotation about the z-axis.
    angle_goal = np.deg2rad(45)
    R_goal = axis_angle_rot_matrix([0, 0, 1], angle_goal)

    # Print initial positions for debugging
    for i in range(arm.N_joints):
        print(f'Joint {i} position = {arm.position(q_0, index=i)}')
    print(f'Initial end-effector position = {arm.position(q_0, index=-1, p_i=p_eff)}')
    print(f'Goal position = {goal_position}')
    print(f'Goal orientation (R_goal):\n{R_goal}')

    # Use the reduced inverse kinematics to obtain joint angles achieving the desired pose.
    final_q = arm.pseudo_inverse_reduced(q_0, p_eff, goal_position, R_goal, max_steps=1000)

    # Print final joint angles in degrees.
    print('\nFinal Joint Angles in Degrees:')
    for i, angle in enumerate(final_q):
        print(f'Joint {i+1}: {np.degrees(angle):.2f}')

if __name__ == '__main__':
    main()
