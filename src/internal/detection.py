import torch
from PIL import ImageDraw as D
from transformers import (
    YolosForObjectDetection,
    YolosFeatureExtractor,
    DetrImageProcessor,
    DetrForObjectDetection,
)
from ultralytics import YOLO
import ultralytics

##########################################################################################
##########################################################################################

# WORK IN PROGRESS
class YoloV8ObjectDetection:
    def __init__(self, th, show):
        print("[EXPERIMENTAL] - [DETECTION] - Using Yolo V8 Object Detection Backend")
        ultralytics.checks()
        self.show = show
        self.th = th
        self.model = YOLO("yolov8n.pt")

    def detect_all_objects(self, image, one_object):
        # Step 1: Detect all objects in image
        object_boxes = self.detect_objects_in_images(image)
        pass

    def detect_objects_in_images(self, image):
        results = self.model.predict(source=image)
        print(len(results))
        results[0].boxes
        return results

    def crop_objects_found(self, image, boxes):
        pass

    def plot_results(self, pil_img, boxes):
        pass


##########################################################################################
##########################################################################################


class YoloV5ObjectDetection:
    def __init__(self, th, show):
        print("[DETECTION] - Using Yolo V5 Object Detection Backend")
        self.show = show
        self.th = th
        self.processor = YolosFeatureExtractor.from_pretrained("hustvl/yolos-small")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    def detect_all_objects(self, image, one_object):
        # Step 1: Detect all objects in image
        object_boxes = self.detect_objects_in_images(image)

        if self.show:
            self.plot_results(image, object_boxes)

        # Step 2: Crop all objects in the image
        objects = self.crop_objects_found(image, object_boxes)

        if one_object:
            print("[DETECTOR] - Set for one_object=True")
            sorted_objects = sorted(objects, key=lambda x: x[3], reverse=True)
            best_object = sorted_objects[0]
            objects = [best_object]
            print(
                f"[DETECTOR] - Returning only best object - Name:{best_object[2]} - Score:{best_object[3]}"
            )

        return objects

    def detect_objects_in_images(self, image):

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.th
        )[0]

        output = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            score = round(score.item(), 3)
            box = [round(i, 2) for i in box.tolist()]
            name = self.model.config.id2label[label.item()]

            print()
            print(f"Detected {name} with confidence " f"{score} at location {box}")
            output.append((box, name, score))

        return output

    def crop_objects_found(self, image, boxes):
        image_objects = []
        for box, name, score in boxes:
            final_box = [max(0, round(b)) for b in box]
            img_object = image.crop(final_box)
            image_objects.append((img_object, final_box, name, score))
        return image_objects

    def plot_results(self, pil_img, boxes):
        img = pil_img.copy()
        for box, _, _ in boxes:
            xmin, ymin, xmax, ymax = box
            draw = D.Draw(img)
            draw.rectangle([(xmin, ymin), (xmax, ymax)])
        img.show()


##########################################################################################
##########################################################################################


class DETRObjectDetection:
    def __init__(self, th, show):
        print("[DETECTION] - Using DETR Object Detection Backend")
        self.show = show
        self.th = th
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def detect_all_objects(self, image, one_object):
        # Step 1: Detect all objects in image
        object_boxes = self.detect_objects_in_images(image)

        if self.show:
            self.plot_results(image, object_boxes)

        # Step 2: Crop all objects in the image
        objects = self.crop_objects_found(image, object_boxes)
        return objects

    def detect_objects_in_images(self, image):

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.th
        )[0]

        output = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            score = round(score.item(), 3)
            box = [round(i, 2) for i in box.tolist()]
            name = self.model.config.id2label[label.item()]

            print()
            print(f"Detected {name} with confidence " f"{score} at location {box}")
            output.append((box, name, score))

        return output

    def crop_objects_found(self, image, boxes):
        image_objects = []
        for box, name, score in boxes:
            final_box = [max(0, round(b)) for b in box]
            img_object = image.crop(final_box)
            image_objects.append((img_object, final_box, name, score))
        return image_objects

    def plot_results(self, pil_img, boxes):
        img = pil_img.copy()
        for box, _, _ in boxes:
            xmin, ymin, xmax, ymax = box
            draw = D.Draw(img)
            draw.rectangle([(xmin, ymin), (xmax, ymax)])
        img.show()
