import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianVariableImpedanceControllerSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from tqdm import trange

import rospy
import argparse
import subprocess

from bagpy import bagreader
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='bag_file')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    print('Reset with joints')
    fa.reset_joints()
    print('Open gripper')
    fa.open_gripper()

    rospy.loginfo('Generating Trajectory')
    dt = 0.01
    T = 5
    ts = np.arange(0, T, dt)
    N = len(ts)

    start_pose = FC.READY_POSE  # NOTE: different start_pose from current pose
    target_poses = [start_pose] * N
    stiffness = FC.DEFAULT_VIC_STIFFNESS
    stiffnesses = [stiffness] * N

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1)
    rate = rospy.Rate(1 / dt)
    n_times = 2

    rospy.loginfo('Sleeping for 1 second')
    rospy.sleep(1)

    rospy.loginfo('Publishing cartesian VIC trajectory...')
    fa.run_dynamic_impedance(duration=T * n_times,
                             buffer_time=T * n_times * 1.5,
                             stiffness=stiffnesses[0])
    for j in range(n_times):
        # Start recording to rosbag
        rosbag_j_fname = str(args.file)+"_"+str(j)
        command = ["rosbag", "record", "-O", rosbag_j_fname, "--duration="+str(T+2),
                   "robot_state_publisher_node_1/robot_state"]
        print(f'Starting rosbag record of: {rosbag_j_fname}')
        rosbag_process = subprocess.Popen(command)

        rospy.sleep(1)

        init_time = rospy.Time.now().to_time()
        for i in trange(N):
            t = i % N
            timestamp = rospy.Time.now().to_time() - init_time

            traj_gen_proto_msg = PosePositionSensorMessage(
                id=i, timestamp=timestamp,
                position=target_poses[t].translation, quaternion=target_poses[t].quaternion
            )
            fb_ctrlr_proto = CartesianVariableImpedanceControllerSensorMessage(
                id=i, timestamp=timestamp,
                stiffness=stiffnesses[t], mass=FC.DEFAULT_VIC_MASS
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_VARIABLE_IMPEDANCE)
            )

            pub.publish(ros_msg)

            rate.sleep()

        # Terminate rosbag-process
        rosbag_process.wait(T)
        rosbag_process.kill()
        #rosbag_process.terminate()

        # Read the rosbag and save the relevant data locally
        ####################################################################################
        directory = "./"  # FIXME: Make sure directory is where the bag files are saved
        extension = ""
        if not rosbag_j_fname.endswith(".bag"):
            extension = ".bag"
        rospy.sleep(3)
        print(f'Reading bag-file: {rosbag_j_fname + extension}')

        b = bagreader(directory + rosbag_j_fname + extension)

        robot_states = b.message_by_topic('robot_state_publisher_node_1/robot_state')
        # TODO: Extract only the ros messages in relevant time span
        df = pd.read_csv(robot_states)

        # Finally, read out the relevant information from the pickle files (or directly from the csv-files??)
        times = df["Time"].to_numpy()
        i_start = (np.abs(times-init_time)).argmin()  # Using data as close to the real robot experiment as possible
        assert i_start+N < len(times)
        times = times[i_start:i_start+N]

        print(f'i_start: {i_start}, times.shape: {times.shape}')

        i_temp = df.columns.get_loc("O_T_EE_0")
        poses = df.iloc[i_start:i_start+N, i_temp:i_temp+16].to_numpy()  # FIXME: or try .values
        print(f'Pose shape before: {poses.shape}')
        poses = poses.reshape((-1, 4, 4))
        print(f'Pose shape after reshaper: {poses.shape}')
        poses = poses.transpose((0, 2, 1))  # FIXME: check if transpose is correct

        i_temp = df.columns.get_loc("O_F_ext_hat_K_0")
        f_ext_hat = df.iloc[i_start:i_start+N, i_temp:i_temp+6].to_numpy()  # FIXME: or try .values

        i_temp = df.columns.get_loc("q_0")
        q = df.iloc[i_start:i_start+N, i_temp:i_temp+7].to_numpy()  # FIXME: or try .values

        i_temp = df.columns.get_loc("dq_0")
        dq = df.iloc[i_start:i_start+N, i_temp:i_temp + 7].to_numpy()  # FIXME: or try .values

        ####################################################################################

        # Print some data form the rosbag to show that it has been collected successfully
        print(f'poses.shape: {poses.shape}, f_ext_hat.shape: {f_ext_hat.shape}, q.shape: {q.shape}')
        print(f'Pose at 5th timestep: {poses[4]}')
        print(f'Joints at 432nd timestep: {q[431]}')
        print(f'Time difference in rosbag_record {times[1]-times[0]}')
        print(f'Duration of rosbag_record {times[-1] - times[0]}')
        print(f'init_time - times[0] {init_time - times[0]}')

    fa.stop_skill()

    rospy.loginfo('Moving to home pose')
    fa.reset_joints()
    rospy.loginfo('Done')

    print(timestamp)
