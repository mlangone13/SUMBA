from src.sumba import Sumba
from PIL import Image


import os
import pyrealsense2 as rs
import numpy as np
import cv2
import time

POSE_FILE           = '/home/guille/catkin_ws/src/arm_controller_g/dest_pose.arm'
GRIPPER_FILE        = '/home/guille/catkin_ws/src/arm_controller_g/gripper_pose.arm'
ARM_SIGNAL_FILE     = '/home/guille/catkin_ws/src/arm_controller_g/arm_signal.arm'
GRIPPER_SIGNAL_FILE = '/home/guille/catkin_ws/src/arm_controller_g/gripper_signal.arm'

def go_to (new_pos):
    f = open(POSE_FILE, 'w')
    data = ""
    for i in new_pos:
        data += str(i) + ","
    f.write(data[:-1])
    f.close()

    originalTime = os.path.getmtime(ARM_SIGNAL_FILE)
    while(os.path.getmtime(ARM_SIGNAL_FILE) == originalTime):
        pass
    time.sleep(0.1)

def move_gripper (pos):
    f = open(GRIPPER_FILE, 'w')
    f.write(str(pos))
    f.close()

    originalTime = os.path.getmtime(GRIPPER_SIGNAL_FILE)
    while(os.path.getmtime(GRIPPER_SIGNAL_FILE) == originalTime):
        pass
    time.sleep(0.1)

sumba = Sumba(
        detector_id="detr",
        detector_th=0.9,
        detector_one_object=False,
        segmentator_id="maskformer",
        grasping_N=50,
        grasping_tol=5,
        show=True,
)


HAND_CAMERA_SN = '037322251488'

# Hand camera configuration
pipeline_hand = rs.pipeline()
config_hand = rs.config()
config_hand.enable_device(HAND_CAMERA_SN)
config_hand.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_hand.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)



cfg = pipeline_hand.start(config_hand)
profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

color_sensor = cfg.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, False)
color_sensor.set_option(rs.option.exposure, 350)
color_sensor.set_option(rs.option.gain, 20)

align_to = rs.stream.color
align = rs.align(align_to)

for i in range (30):

    frames = pipeline_hand.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    print ("deleting img", i)
# if not depth_frame or not color_frame:
#     print ('No hand frame')
#     return True





depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

img = Image.fromarray(color_image[:,:,::-1])


points = sumba.run_pipeline(img) 
print (points)


OBJECT_HEIGHT = 8
DEPTH_HARDCODED = 52 - OBJECT_HEIGHT

DEFAULT_HEIGHT = 535

for o in points:

    x, y, x1, y1 = o[1]
    ang = np.degrees(np.arctan(abs(y-y1)/abs(x-x1)))
    
    if x > x1:
        print ("invirtiendo")
        ang *= -1

    cx, cy = int((x+x1)/2), int((y+y1)/2)

    x, y, x1, y1 = o[2]

    sliced_d = depth_image[y:y1, x:x1]

    obj_d   = np.min(sliced_d[sliced_d != 0])
    table_d = min(np.max(sliced_d), DEFAULT_HEIGHT)

    print ("Trying to grasp", o[0])
    print (table_d, obj_d)
    print ("object height:", table_d-obj_d)

    d_f = np.mean([table_d, obj_d])/1000

    result = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], d_f)
    pose = list(result) + [ang, 70, 0]

    above_pose = pose.copy()
    above_pose[2] -= 0.10
    go_to(above_pose)
    
    go_to(pose)

    move_gripper(1)

    go_to([0, -0.3, 0.3, 0, 70, 0])

    go_to([-0.3, -0.45, 0.2, 0, 50, -40])

    go_to([-0.3, -0.45, 0.4, 0, 50, -40])

    move_gripper(0)

    go_to([0, -0.3, 0.3, 0, 70, 0])
