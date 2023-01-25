import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianVariableImpedanceControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import transform_to_list, min_jerk, convert_rigid_transform_to_array

from tqdm import trange

import rospy

import argparse
import pickle as pkl
import scipy.io


def create_formated_dict(target_poses, stiffnesses, poses, joints, joint_torques, joint_velocities, ee_force_torques,
                         times):
    state_dict = {}
    state_dict['target_poses'] = np.array(target_poses)
    state_dict['stiffnesses'] = np.array(stiffnesses)
    state_dict['poses'] = np.array(poses)
    state_dict['q'] = np.array(joints)
    state_dict['tau_ext'] = np.array(joint_torques)
    state_dict['dq'] = np.array(joint_velocities)
    state_dict['f_ext_hat'] = np.array(ee_force_torques)
    state_dict['times'] = np.array(times)

    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='franka_traj.pkl')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    print('Reset with joints')
    fa.reset_joints()
    print('Open gripper')
    fa.open_gripper()

    '''
    # For drawing task
    input('Press [Enter] to close grippers after 3 seconds.')
    rospy.loginfo('Sleeping for 3 second')
    rospy.sleep(3)
    rospy.loginfo('Closing grippers')
    fa.goto_gripper(0.01, grasp=True, speed=0.02, force=20.0)
    
    while True:
        input('Press [Enter] to enter guide mode and move robot to be on top of a flat surface.')
        fa.run_guide_mode()
        while True:
            inp = input('[r]etry or [c]ontinue? ')
            if inp not in ('r', 'c'):
                print('Please give valid input!')
            else:
                break
        if inp == 'c':
            break
    '''

    rospy.loginfo('Generating Trajectory')
    # EE will follow a 2D circle from the home position
    dt = 0.01
    T = 10
    ts = np.arange(0, T, dt)
    N = len(ts)
    dthetas = np.linspace(-np.pi / 2, 3 * np.pi / 2, N)
    r = 0.07
    circ = r * np.c_[np.sin(dthetas), np.cos(dthetas)]

    start_pose = fa.get_pose()

    #start_pose.rotation = FC.HOME_POSE.rotation
    #rospy.loginfo('Moving to right orientation')
    #fa.goto_pose(start_pose, ignore_virtual_walls=True)

    target_poses = []
    for i, t in enumerate(ts):
        pose = start_pose.copy()
        pose.translation[0] += r + circ[i, 0]
        pose.translation[1] += circ[i, 1]
        #pose.translation[2] -= 0.02
        target_poses.append(pose)
    stiffness = FC.DEFAULT_VIC_STIFFNESS
    #stiffness = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    mass = FC.DEFAULT_VIC_MASS

    stiffness_gain = np.ones(N) #np.linspace(2, 0.1, N)
    stiffnesses = []
    for i, t in enumerate(ts):
        stiffness_i = [stiffness_gain[i] * stiffness[j] for j in range(len(stiffness))]
        stiffnesses.append(stiffness_i)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)
    n_times = 1

    i_publish = 10;  # publish every i_publish time step

    # For saving state trajectory:
    poses = []
    joints = []
    joint_torques = []
    joint_velocities = []
    ee_force_torques = []
    times = []

    rospy.loginfo('Sleeping for 1 second')
    rospy.sleep(1)

    rospy.loginfo('Publishing cartesian VIC trajectory...')
    fa.run_dynamic_impedance(duration=T * n_times,
                             buffer_time=T * n_times * 1.5,
                             stiffness=stiffnesses[0],
                             mass=mass)
    init_time = rospy.Time.now().to_time()
    for i in trange(N * n_times):
        t = i % N
        timestamp = rospy.Time.now().to_time() - init_time

        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp,
            position=target_poses[t].translation, quaternion=target_poses[t].quaternion
        )
        fb_ctrlr_proto = CartesianVariableImpedanceControllerSensorMessage(
            id=i, timestamp=timestamp,
            stiffness=stiffnesses[t], mass=mass
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_VARIABLE_IMPEDANCE)
        )

        pub.publish(ros_msg)

        # Save current state
        if i % i_publish == 0:
            pose_array = convert_rigid_transform_to_array(fa.get_pose())
            poses.append(pose_array)
            joints.append(fa.get_joints())
            joint_torques.append(fa.get_joint_torques())
            ee_force_torques.append(fa.get_ee_force_torque())
            times.append(timestamp)
        rate.sleep()
    fa.stop_skill()

    rospy.loginfo('Moving to home pose')
    fa.reset_joints()
    # rospy.loginfo('Opening Gripper')
    # fa.open_gripper()

    rospy.loginfo('Done')

    state_dict = create_formated_dict(target_poses, stiffnesses, poses, joints, joint_torques, joint_velocities,
                                      ee_force_torques, times)

    # Save state dict as .pkl and .mat file
    with open(args.file + '.pkl', 'wb') as pkl_f:
        pkl.dump(state_dict, pkl_f)
        print("Did save skill dict: {}".format(args.file))

    scipy.io.savemat(args.file + '.mat', state_dict, oned_as='row')

    print(timestamp)
