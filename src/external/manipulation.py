import os
import time

POSE_FILE = "/home/guille/catkin_ws/src/arm_controller_g/dest_pose.arm"
GRIPPER_FILE = "/home/guille/catkin_ws/src/arm_controller_g/gripper_pose.arm"
ARM_SIGNAL_FILE = "/home/guille/catkin_ws/src/arm_controller_g/arm_signal.arm"
GRIPPER_SIGNAL_FILE = "/home/guille/catkin_ws/src/arm_controller_g/gripper_signal.arm"


def go_to(new_pos):
    f = open(POSE_FILE, "w")
    data = ""
    for i in new_pos:
        data += str(i) + ","
    f.write(data[:-1])
    f.close()

    time.sleep(0.01)
    originalTime = os.path.getmtime(ARM_SIGNAL_FILE)
    while os.path.getmtime(ARM_SIGNAL_FILE) == originalTime:
        pass
    time.sleep(0.1)


def move_gripper(pos):
    f = open(GRIPPER_FILE, "w")
    f.write(str(pos))
    f.close()

    time.sleep(0.1)
    originalTime = os.path.getmtime(GRIPPER_SIGNAL_FILE)
    while os.path.getmtime(GRIPPER_SIGNAL_FILE) == originalTime:
        pass
    time.sleep(0.1)


class Manipulator:
    def __init__(self):

        self.open_gripper = 0
        self.close_gripper = 1
        self.home = [0, -0.3, 0.3, 0, 70, 0]
        self.upper_box = [-0.3, -0.45, 0.2, 0, 50, -40]
        self.lower_box = [-0.3, -0.45, 0.4, 0, 50, -40]
        pass

    def execute_manipulation(self, result, ang):
        pose = list(result) + [ang, 60, 0]
        above_pose = pose.copy()
        above_pose[2] -= 0.10

        go_to(above_pose)

        go_to(pose)

        move_gripper(self.close_gripper)

        go_to(self.home)

        go_to(self.upper_box)

        go_to(self.lower_box)

        move_gripper(self.open_gripper)

        go_to(self.upper_box)

        go_to(self.home)
