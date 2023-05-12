import pyrealsense2 as rs
import numpy as np
from PIL import Image


class Camera:
    def __init__(self):

        self.OBJECT_HEIGHT = 8
        self.DEPTH_HARDCODED = 52 - self.OBJECT_HEIGHT
        self.DEFAULT_HEIGHT = 535

        HAND_CAMERA_SN = "037322251488"

        # Hand camera configuration
        self.pipeline_hand = rs.pipeline()
        config_hand = rs.config()
        config_hand.enable_device(HAND_CAMERA_SN)
        config_hand.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config_hand.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        cfg = self.pipeline_hand.start(config_hand)
        profile = cfg.get_stream(
            rs.stream.color
        )  # Fetch stream profile for depth stream
        self.intr = (
            profile.as_video_stream_profile().get_intrinsics()
        )  # Downcast to video_stream_profile and fetch intrinsics

        color_sensor = cfg.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, False)
        color_sensor.set_option(rs.option.exposure, 350)
        color_sensor.set_option(rs.option.gain, 20)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.depth_frame = None
        self.color_frame = None

    def discard_frames(self, frames=30):
        for i in range(30):
            frames = self.pipeline_hand.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.depth_frame = aligned_frames.get_depth_frame()
            self.color_frame = frames.get_color_frame()
            print("deleting img", i)

    def get_frame(self):
        frames = self.pipeline_hand.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(self.depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())

        return color_image, depth_image

    def change_channel(self, color_image):
        img = Image.fromarray(color_image[:, :, ::-1])

    def project_points(self, object, depth_image):
        x, y, x1, y1 = object[1]
        cx, cy = int((x + x1) / 2), int((y + y1) / 2)
        ang = np.degrees(np.arctan(abs(y - y1) / abs(x - x1)))

        if x > x1:
            ang *= -1

        x, y, x1, y1 = object[2]
        sliced_d = depth_image[y:y1, x:x1]

        obj_d = np.min(sliced_d[sliced_d != 0])
        table_d = min(np.max(sliced_d), self.DEFAULT_HEIGHT)

        print("Trying to grasp", o[0])
        print(table_d, obj_d)
        print("Object height:", table_d - obj_d)

        d_f = np.mean([table_d, obj_d]) / 1000

        result = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], d_f)

        return result, ang
