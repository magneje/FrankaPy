import subprocess
import time
if __name__ == "__main__":
    command = ["rosbag", "record", "test.bag"]
    '''rosbag_process = subprocess.Popen(command)
    try:
        rosbag_process.wait(3)
    except subprocess.TimeoutExpired:
        rosbag_process.kill()
        print('error: TimeoutExpired')'''
    rosbag_process = subprocess.Popen(["exit", "--help"])
    try:
        rosbag_process.wait(3)
    except subprocess.TimeoutExpired:
        rosbag_process.kill()
        print('error: TimeoutExpired')
    proc_1 = subprocess.Popen(["echo", "hello world"])
    proc_2 = subprocess.Popen(["rosbag", "--help"])
    proc_2.wait(3)
