from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianVariableImpedanceControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from tqdm import trange, tqdm

import rospy
import argparse
import pickle as pkl
# import scipy.io
import subprocess

from bagpy import bagreader
import pandas as pd

import numpy as np

from utilities import demonstrate_trajectory
from frankapy.utils import franka_pose_to_rigid_transform

import matplotlib.pyplot as plt

import os
import quaternion


def is_task_completed(pose_final, target_pose_final, he_hat_final, e_xy_max, e_z_max, f_z_max):
    """
    requirements: [e_xy, e_z, f_z]
        e_xy_max: max (Euclidian) distance from goal in xy-plane
        e_z_max: max distance from goal in z-direction
        f_z_max: max (due to sign) external force in z-direction
    """
    # TODO: consider adding requirement for orientation (e.g. theta < 10 degrees)
    e_pos = pose_final.position - (target_pose_final.position - delta_xd_final)
    e_xy_squared = e_pos[0] * e_pos[0] + e_pos[1] * e_pos[1]
    e_z = e_pos[2]
    f_ext_z = he_hat_final[2]

    is_success_ = False

    if e_xy_squared <= e_xy_max*e_xy_max and abs(e_z) <= e_z_max and f_ext_z <= f_z_max:
        print(f'\te_xy = {np.sqrt(e_xy_squared)} <= e_xy_max = {e_xy_max}')
        print(f'\t|e_z = {e_z}| <= e_z_max = {e_z_max}')
        print(f'\tf_ext_z = {f_ext_z} <= f_z_max = {f_z_max}')
        is_success_ = True
    else:
        print('Task FAILED because:')
        if e_xy_squared >= e_xy_max*e_xy_max:
            print(f'\te_xy = {np.sqrt(e_xy_squared)} > e_xy_max = {e_xy_max}')
        if abs(e_z) >= e_z_max:
            if e_z < 0:
                txt = "below"
            else:
                txt = "above"
            print(f'\t|e_z = {e_z}| > e_z_max = {e_z_max} (manipulator stopped {txt} target position)')
        if f_ext_z >= f_z_max:
            print(f'\tf_ext_z = {f_ext_z} > f_z_max = {f_z_max}')
        is_success_ = False

    while True:
        is_success_override = input(
            f'Task success: {is_success_}. Press [Enter] to continue or override by typing 1 (success) or 0 (fail)?: ')
        if is_success_override == '':
            break
        elif is_success_override not in ('1', '0'):
            print('Please give valid input!')
        else:
            is_success_ = is_success_override
            break
    return is_success_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', default='')
    args = parser.parse_args()

    # Experiment parameters:
    save_states = True
    add_validation_trajectory = True
    dt = 0.01
    T_demo = 8.
    duplo_width = 0.04
    n_demos = 3
    n_demos += 1 if add_validation_trajectory else 0
    T_insert = 1.  # The time allocated for final insertion phase
    T_converge = 1.  # The time allocated for convergence after insertion
    duration = T_demo + T_insert + T_converge
    rotational_stiffness = [30.] * 3
    demo_trans_stiffness = [800.] * 3
    e_xy_max = 0.1*duplo_width  # [m]
    e_z_max = 0.001  # [m]
    f_z_max = -1.  # [N]
    delta_xd_final = [0., 0., -0.03]  # Final target position relative final position, used to generate downwards force required for insertion

    # If folder doesn't exist, then create it.
    folder_exists = os.path.isdir(args.dir) or args.dir == ''
    if not folder_exists:
        os.makedirs(args.dir)
        print("Created folder : ", args.dir)
    else:
        print(args.dir, "folder already exists. Will overwrite data")
        input('Press [Enter] to continue or [CTRL-C] to exit program')

    print('Save states set to {}'.format(save_states))
    data_dict = {'demo': {}, 'experimental_setup': {}}

    # dtype = torch.float32

    print('Starting robot')
    fa = FrankaArm()
    print('Reset with joints')
    fa.reset_joints()

    target_poses = [demonstrate_trajectory(fa, T_demo, dt, duplo_width) for j in range(n_demos)]
    ts = np.arange(0, duration, dt)
    N = len(ts)
    N_insert = round(T_insert / dt)
    delta_target_positions = np.c_[np.linspace(0, delta_xd_final[0], N_insert),
                                   np.linspace(0, delta_xd_final[1], N_insert),
                                   np.linspace(0, delta_xd_final[2], N_insert)]

    assert len(target_poses) == n_demos, f"len(target_poses) = {len(target_poses)}"  # TODO: remove when checked once

    # Add final insertion phase to the end of desired trajectories
    for j in range(n_demos):
        base_pose = target_poses[j][-1].copy()
        for i in range(N_insert):
            pose = base_pose.copy()
            pose.translation += delta_target_positions[i]
            target_poses[j].append(pose)
        base_pose = target_poses[j][-1].copy()
        for i in range(len(target_poses[j]), N):
            target_poses[j].append(base_pose.copy())

    for j, pose in enumerate(target_poses):
        assert len(pose) == N, f"len(pose_{j}) = {len(pose)}"

    target_velocities = np.zeros((N * n_demos, 6))  # TODO: define target_velocities (for now 0)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)

    ##### 1) Perform trajectories with constant (high) impedances ######################################################
    for j in range(n_demos):
        # Compute desired trajectory
        target_poses_j = target_poses[j]
        target_positions_j = np.zeros((N, 3))
        target_quaternions_j = np.zeros((N, 4))
        for i in range(N):
            target_positions_j[i] = target_poses_j[i].translation
            target_quaternions_j[i] = target_poses_j[i].quaternion
        xd_j = np.concatenate((target_positions_j, target_quaternions_j[:, 1:]),
                              axis=1)  # desired trajectory for current demo-run
        vd_j = target_velocities[j * N:(j + 1) * N]

        demo_stiffness_ = demo_trans_stiffness + rotational_stiffness
        print(f'Demo stiffness is {demo_stiffness_}')

        rospy.loginfo(f'Go to starting pose for demo {j}')
        fa.goto_pose(target_poses_j[0], ignore_virtual_walls=True)
        rospy.loginfo('Initialising constant impedance control')
        fa.run_dynamic_impedance(duration=duration,
                                 buffer_time=3,
                                 stiffness=demo_stiffness_)
        # Start recording to rosbag
        rosbag_j_fname = args.dir + '/' + "demo_" + str(j)
        command = ["rosbag", "record", "-O", rosbag_j_fname, "--duration=" + str(duration + 2),
                   "robot_state_publisher_node_1/robot_state", "franka_ros_interface/sensor"]
        rospy.loginfo(f'Starting rosbag record of: {rosbag_j_fname}')
        rosbag_process = subprocess.Popen(command)
        rospy.sleep(1)  # Sleeep to give subprocess time to start
        rospy.loginfo('Publishing trajectory with constant impedance control...')

        # Execute trajectory with constant stiffness
        init_time = rospy.Time.now().to_time()
        for i in trange(N):
            timestamp = rospy.Time.now().to_time() - init_time
            traj_gen_proto_msg = PosePositionSensorMessage(
                id=i, timestamp=timestamp,
                position=target_positions_j[i], quaternion=target_quaternions_j[i]
            )
            fb_ctrlr_proto = CartesianVariableImpedanceControllerSensorMessage(
                id=i, timestamp=timestamp, stiffness=demo_stiffness_
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_VARIABLE_IMPEDANCE)
            )
            pub.publish(ros_msg)
            rate.sleep()
        fa.stop_skill()
        print(f'Timestamp = {timestamp}')  # TODO: remove unnecessary prints

        # Terminate rosbag-process
        rospy.loginfo(f'Waiting on rosbag process...')
        rosbag_process.wait(duration)
        rosbag_process.kill()

        # Load relevant data from rosbag ###################################################  # TODO: consider wrapping around a function
        extension = ""
        if not rosbag_j_fname.endswith(".bag"):
            extension = ".bag"
        rospy.loginfo('Sleeping for 2 seconds while bag data is being saved')
        rospy.sleep(2)
        rospy.loginfo(f'Loading bag-file: {rosbag_j_fname + extension}')
        b = bagreader(rosbag_j_fname + extension)

        robot_states = b.message_by_topic('robot_state_publisher_node_1/robot_state')
        df = pd.read_csv(robot_states)

        times = df["Time"].to_numpy()
        i_start = (np.abs(times - init_time)).argmin()  # Using data as close to the real robot experiment as possible
        assert i_start + N < len(times)
        times = times[i_start:i_start + N] - init_time

        j_temp = df.columns.get_loc("O_T_EE_0")
        poses_data = df.iloc[i_start:i_start + N, j_temp:j_temp + 16].to_numpy()
        assert poses_data.shape[0] == N
        poses = []
        for i in range(N):
            poses.append(franka_pose_to_rigid_transform(poses_data[i].copy()))

        positions = np.zeros((N, 3))
        quaternions = np.zeros((N, 4))
        for i in range(N):
            positions[i] = poses[i].translation
            quaternions[i] = poses[i].quaternion

        xe = np.concatenate((positions, quaternions[:, 1:]), axis=1)

        j_temp = df.columns.get_loc("O_F_ext_hat_K_0")
        he_hat = df.iloc[i_start:i_start + N, j_temp:j_temp + 6].to_numpy()

        j_temp = df.columns.get_loc("q_0")
        q = df.iloc[i_start:i_start + N, j_temp:j_temp + 7].to_numpy()

        j_temp = df.columns.get_loc("dq_0")
        dq = df.iloc[i_start:i_start + N, j_temp:j_temp + 7].to_numpy()

        j_temp = df.columns.get_loc("tau_J_d_0")
        tau_J_d = df.iloc[i_start:i_start + N, j_temp:j_temp + 7].to_numpy()  # without gravity compensation
        ####################################################################################

        # Determine if success or fail before resetting
        is_success = is_task_completed(poses[-1], target_poses_j[-1], he_hat[-1], e_xy_max, e_z_max, f_z_max)

        rospy.loginfo('Reset with joints')
        fa.reset_joints()

        # Compute ve from dq and jacobian()
        ve = np.zeros((N, 6))
        for i in range(N):
            J_i = fa.get_jacobian(q[i])
            ve[i] = J_i @ dq[i]

        # Input data X
        # X_j = torch.tensor(np.concatenate((xe, ve, he_hat, xd_j, vd_j), axis=1), dtype=dtype)
        # X = X_j if j == 0 else torch.cat((X, X_j), 0)

        # Compute cartesian end effector control wrench (force, torque) hc
        hc = np.zeros((N, 6))
        for i in range(N):
            J_i = fa.get_jacobian(q[i])
            hc[i] = np.linalg.pinv(J_i.T) @ tau_J_d[i]

        # True labels Y
        # Y_j = torch.tensor(hc[:, :3], dtype=dtype)
        # Y = Y_j if j == 0 else torch.cat((Y, Y_j), 0)

        # Compute angle-axis from quaternions
        angle_axis_j = np.array(quaternion.as_rotation_vector(quaternion.as_quat_array(quaternions.copy())))
        target_angle_axis_j = np.array(quaternion.as_rotation_vector(quaternion.as_quat_array(target_quaternions_j.copy())))

        # Save data to data_dict: [times, poses, positions, quaternions, target_poses, target_positions,
        #                          target_quaternions, ve, vd, he_hat,  joints, is_success]
        data_dict_j = {'times': times, 'poses': poses, 'positions': positions, 'quaternions': quaternions,
                       'target_poses': target_poses_j, 'target_positions': target_positions_j,
                       'target_quaternions': target_quaternions_j, 've': ve, 'vd': vd_j, 'he_hat': he_hat,
                       'hc': hc, 'joints': q, 'is_success': is_success, 'angle_axis': angle_axis_j,
                       'target_angle_axis': target_angle_axis_j}

        data_dict['demo'][j] = data_dict_j

    # Plot nominal trajectories
    plt.figure()
    plt.subplot(311)
    legend_list = []
    for j in range(n_demos-int(add_validation_trajectory)):
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 0],
                 ts, data_dict['demo'][j]['target_positions'][:, 0], '--')
        legend_list.extend([f'Actual (demo {j+1})', f'Desired (demo {j + 1}'])
    plt.legend(legend_list)
    plt.title('Trajectory x')
    plt.ylabel(r'$x$ [m]')
    plt.subplot(312)
    legend_list = []
    for j in range(n_demos-int(add_validation_trajectory)):
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 1],
                 ts, data_dict['demo'][j]['target_positions'][:, 1], '--')
        legend_list.extend([f'Actual (demo {j+1})', f'Desired (demo {j + 1}'])
    plt.legend(legend_list)
    plt.title('Trajectory y')
    plt.ylabel(r'$y$ [m]')
    plt.subplot(313)
    legend_list = []
    for j in range(n_demos-int(add_validation_trajectory)):
        plt.plot(data_dict['demo'][j]['times'], data_dict['demo'][j]['positions'][:, 2],
                 ts, data_dict['demo'][j]['target_positions'][:, 2], '--')
        legend_list.extend([f'Actual (demo {j+1})', f'Desired (demo {j + 1}'])
    plt.legend(legend_list)
    plt.title('Trajectory z')
    plt.ylabel(r'$z$ [m]')
    plt.xlabel('Time [s]')
    plt.suptitle('Actual and desired demonstration trajectories')
    plt.savefig(args.dir + '/position_demo.png', bbox_inches='tight')
    plt.show()
    if add_validation_trajectory:
        plt.figure()
        plt.subplot(311)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 0],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 0], '--')
        plt.legend(['Actual', 'Desired'])
        plt.title('Trajectory x')
        plt.ylabel(r'$x$ [m]')
        plt.subplot(312)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 1],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 1], '--')
        plt.legend(['Actual', 'Desired'])
        plt.title('Trajectory y')
        plt.ylabel(r'$y$ [m]')
        plt.subplot(313)
        plt.plot(data_dict['demo'][n_demos - 1]['times'], data_dict['demo'][n_demos - 1]['positions'][:, 2],
                 ts, data_dict['demo'][n_demos - 1]['target_positions'][:, 2], '--')
        plt.legend(['Actual', 'Desired'])
        plt.title('Trajectory z')
        plt.ylabel(r'$z$ [m]')
        plt.xlabel('Time [s]')
        plt.suptitle('Actual and desired validation trajectory')
        plt.savefig(args.dir + '/position_validation.png', bbox_inches='tight')
        plt.show()

    # Add useful parameters to data_dict
    data_dict['experimental_setup']['dt'] = dt
    data_dict['experimental_setup']['duplo_width'] = duplo_width
    data_dict['experimental_setup']['duration'] = duration
    data_dict['experimental_setup']['T_demo'] = T_demo
    data_dict['experimental_setup']['rotational_stiffness'] = rotational_stiffness
    data_dict['experimental_setup']['demo_trans_stiffness'] = demo_trans_stiffness

    # Save data dict to file (.pkl and .mat)
    if save_states:
        with open(args.dir + '/data_dict.pkl', 'wb') as pkl_f:
            pkl.dump(data_dict, pkl_f)
            print(f'Did save data dict: {args.dir}/data_dict.pkl')
