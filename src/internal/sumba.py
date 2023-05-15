import numpy as np
from PIL import Image
import random as rng
import cv2

from src.internal.detection import (
    YoloV5ObjectDetection,
    YoloV8ObjectDetection,
    DETRObjectDetection,
)
from src.internal.segmentation import MaskFormerSegmentation, YoloV8Segmentation
from src.internal.grasping import GraspDetection
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

##########################################################################################
##########################################################################################


class Sumba:
    def __init__(
        self,
        detector_id="yolov5",
        detector_th=0.9,
        detector_max_object_size=0.5,
        segmentator_id="maskformer",
        segmentator_min_mask_size=0.5,
        grasping_N=50,
        grasping_tol=5,
        show=True,
    ):

        rng.seed(12345)

        self.show = show

        if detector_id == "yolov5":
            self.detector = YoloV5ObjectDetection(
                detector_th, detector_max_object_size, show
            )
        elif detector_id == "yolov8":
            self.detector = YoloV8ObjectDetection(
                detector_th, detector_max_object_size, show
            )
        elif detector_id == "detr":
            self.detector = DETRObjectDetection(
                detector_th, detector_max_object_size, show
            )
        else:
            self.show_valid_models()

        if segmentator_id == "maskformer":
            self.segmentator = MaskFormerSegmentation(segmentator_min_mask_size, show)
        elif segmentator_id == "yolov8":
            self.segmentator = YoloV8Segmentation(segmentator_min_mask_size, show)
        else:
            self.show_valid_models()

        if (grasping_N < 10 or grasping_N > 100) or (
            grasping_tol < 5 or grasping_tol > 50
        ):
            self.show_valid_models()
        else:
            self.grasp = GraspDetection(show, grasping_N, grasping_tol)

    def show_valid_models(self):
        print(
            """
        Please select any of the following options for our pipeline -->

        Detector Module:
            - detector_id = ['yolov5','yolov8','detr']
            - detector_th = [0...1]
            - detector_max_object_size = [0...1]

        Segmentator Module:
            - segmentator_id = ['yolov8', 'maskformer']
            - segmentator_min_mask_size = [0...1]

        Grasping Module:
            - grasping_N   = [10...100]
            - grasping_tol = [5 ...50]
        """
        )
        raise Exception("Please select valid model configuration")

    def run_pipeline(self, image, detector_one_object):

        # Step 0: Initizalize
        print("")
        print("***************************************************************")
        print("**************** STARTING OBJECT RECO PIPELINE ****************")
        print("***************************************************************")
        results = []

        if self.show:
            image.show()

        # ------------------------------------------------------------
        # -------------------- OBJECT DETECTION
        # ------------------------------------------------------------
        objects = self.detector.detect_all_objects(image, detector_one_object)

        for object_raw in objects:

            if self.show:
                print("")
                print("--- New Object Detected ---")

            try:
                first_object = object_raw[0]
                first_box = object_raw[1]

                # ------------------------------------------------------------
                # -------------------- OBJECT SEGMENTATION
                # ------------------------------------------------------------
                src_gray, label = self.segmentator.segment_object(first_object)

                # ------------------------------------------------------------
                # -------------------- GRASP DETECTOR
                # ------------------------------------------------------------

                best_absolut_grasp = self.grasp.find_grasping_point(src_gray, first_box)

                # ------------------------------------------------------------
                # -------------------- CONCAT RESULTS
                # ------------------------------------------------------------

                results.append((label, best_absolut_grasp, first_box))
            except Exception as e:
                print(f"!!! ------------ Error ------------ !!!")
                print(f" Exception: {e}")
                print("")
                #raise(e)

        # ------------------------------------------------------------
        # -------------------- FINAL VISUALIZATION
        # ------------------------------------------------------------
        self.show_final_results(image, results)

        return results

    def show_final_results(self, image, results):
        if self.show:
            out = np.array(image)
            for result in results:
                r = [int(x) for x in result[1]]
                cv2.circle(out, (r[0], r[1]), 5, (255, 0, 0), -1)
                cv2.circle(out, (r[2], r[3]), 5, (255, 0, 0), -1)
            img_end = Image.fromarray(out)
            img_end.show()
