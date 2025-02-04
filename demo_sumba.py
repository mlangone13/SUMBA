from src.internal.sumba import Sumba
from src.external.manipulation import Manipulator
from src.external.camera import Camera


def print_initial_configuration():
    print(
        """
        Press the following key to continue

        1 = Run pipeline with one image
        2 = Run pipeline with multiple image
        q = Exit
        
        """
    )


def run_demo(manipulator, camera, sumba):
    print_initial_configuration()
    input_value = input()
    while input_value != "q":

        if str(input_value) == "1":
            print(" ******* Running pipeline with 1 image taken *******")

            color_image, depth_image = camera.get_frame()
            img = camera.change_channel(color_image)

            points = sumba.run_pipeline(img, False)
            for object in points:
                result, ang = camera.project_points(object, depth_image)
                manipulator.execute_manipulation(result, ang)

        elif str(input_value) == "2":

            print(" ******* Running pipeline with multiple image taken *******")

            more_objects = True

            while more_objects:

                color_image, depth_image = camera.get_frame()
                img = camera.change_channel(color_image)

                points = sumba.run_pipeline(img, True)

                for object in points:
                    result, ang = camera.project_points(object, depth_image)
                    manipulator.execute_manipulation(result, ang)

                more_objects = len(points) > 0

        else:
            print(" Select another key to continue...")

        print_initial_configuration()
        input_value = input()



if __name__ == "__main__":
    manipulator = Manipulator()
    camera = Camera()
    sumba = Sumba(
        detector_id="yolov8",
        detector_th=0.05,
        detector_max_object_size=0.1,
        segmentator_id="yolov8",
        segmentator_min_mask_size=0.3,
        grasping_N=50,
        grasping_tol=5,
        show=True,
    )
    camera.discard_frames(30)

    run_demo(manipulator, camera, sumba)
