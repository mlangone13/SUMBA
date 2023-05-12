from src.sumba import Sumba
from src.manipulation import Manipulator
from src.camera import Camera


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

            have_objects = True

            while have_objects:

                color_image, depth_image = camera.get_frame()
                img = camera.change_channel(color_image)

                points = sumba.run_pipeline(img, True)

                for object in points:
                    result, ang = camera.project_points(object, depth_image)
                    manipulator.execute_manipulation(result, ang)

                have_objects = len(points) > 0

        else:
            print(" Select another key to continue...")

        print_initial_configuration()


if __name__ == "__main__":
    manipulator = Manipulator()
    camera = Camera()
    sumba = Sumba(
        detector_id="detr",
        detector_th=0.8,
        detector_one_object=False,
        segmentator_id="maskformer",
        grasping_N=50,
        grasping_tol=5,
        show=True,
    )
    camera.discard_frames(30)

    run_demo(manipulator, camera, sumba)
