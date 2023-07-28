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

from autolab_core import RigidTransform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T


def testNeuralNet():
    dtype = torch.float32
    input_size = 10
    output_size = 5
    x = torch.zeros((64, input_size), dtype=dtype)  # batch_size 64
    model = nn.Sequential(
        nn.Linear(input_size, 100),
        nn.ReLU(),
        nn.Linear(100, output_size),
        nn.Sigmoid()
    )
    scores = model(x)
    print(scores.size())  # you should see [64,5]
    #print(scores)  # you should see values in range [0, 1]

if __name__ == "__main__":

    print('Initialising Pytorch')
    dtype = torch.float32
    USE_GPU = True
    device = torch.device('cuda') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    print('Testing neural network')
    testNeuralNet()

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

    start_pose = fa.get_pose()
    target_poses = [start_pose] * N
    stiffness = FC.DEFAULT_VIC_STIFFNESS
    stiffnesses = [stiffness] * N

    ################################################################
    # Temp: overwrite desired trajectory and stiffnesses
    '''
    with open("../Data/data_force_task_1.pkl", 'rb') as f:
        data = pkl.load(f)
    #stiffnesses = data['stiffnesses']
    #stiffnesses = np.maximum(20, stiffnesses)
    for i in range(len(stiffnesses)):
        target_poses[i].translation = data['target_positions'][i]

    print('Close gripper')
    fa.close_gripper(grasp=False)
    '''
    # Temp end
    ################################################################

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
        rospy.loginfo('Finished a run, testing neural network')
        testNeuralNet()

    fa.stop_skill()

    rospy.loginfo('Moving to home pose')
    fa.reset_joints()
    rospy.loginfo('Done')

    print(timestamp)
