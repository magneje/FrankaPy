import time
from frankapy import FrankaArm
import rospy
from frankapy import FrankaConstants as FC
from frankapy.utils import convert_rigid_transform_to_array
import numpy as np


def demonstrate_trajectory(fa, duration, dt, gripping_width=0.04):
    n = round(duration / dt)
    while True:
        inp = input('Press [Enter] to enter guide mode for 5s and move end effector to desired starting pose '
              'or [c]ontinue to start from this pose: ')
        if inp == '':
            fa.run_guide_mode(5)
        inp = input('Press [Enter] to open gripper then grasp (after 3s wait) or [c]ontinue to leave gripper as it is: ')
        if inp == '':
            rospy.sleep(3)
            print('Open gripper')
            if fa.get_gripper_width() < gripping_width + 0.01:
                fa.open_gripper()
            print('Closing gripper')
            fa.goto_gripper(gripping_width, grasp=True, force=FC.GRIPPER_MAX_FORCE)
        input(f'Press [Enter] to record trajectory for {duration}s (after 3s wait).')
        end_effector_poses = []
        rospy.sleep(3)
        print(f'Applying 0 force torque control for {duration}s')
        fa.apply_effector_forces_torques(duration, 0, 0, 0, block=False, buffer_time=duration, skill_desc='GuideMode')
        init_time = rospy.Time.now().to_time()
        for i in range(n):
            end_effector_poses.append(fa.get_pose())
            rospy.sleep(dt)
        fa.stop_skill()
        timestamp = rospy.Time.now().to_time() - init_time
        print(f'Timestamp: {timestamp}')
        while True:
            inp = input('[r]etry demonstration or [c]ontinue? ')
            if inp not in ('r', 'c'):
                print('Please give valid input!')
            else:
                break
        print('Reset with joints')
        fa.reset_joints()
        if inp == 'c':
            break
    return end_effector_poses
