#%matplotlib inline
#%matplotlib notebook

import torch
import numpy as np
from PIL import Image, ImageDraw as D
from transformers import (
    YolosForObjectDetection,
    AutoFeatureExtractor,
)

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

##########################################################################################
##########################################################################################


class YoloObjectDetection:
    def __init__(self, show):
        self.show = show
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "hustvl/yolos-small"
        )
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    def detect_all_objects(self, image):
        # Step 1: Detect all objects in image
        probas, keep, bboxes_scaled = self.detect_objects_in_images(image)

        if self.show:
            self.plot_results(image, probas[keep], bboxes_scaled[keep])

        # Step 2: Crop all objects in the image
        objects = self.crop_objects_found(image, np.array(bboxes_scaled[keep]))
        return objects

    def detect_objects_in_images(self, image):

        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = self.model(pixel_values, output_attentions=True)

        # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.8

        # rescale bounding boxes
        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = self.feature_extractor.post_process(
            outputs, target_sizes
        )
        bboxes_scaled = postprocessed_outputs[0]["boxes"]

        return probas, keep, bboxes_scaled

    def crop_objects_found(self, image, boxes):
        image_objects = []
        for box in boxes:
            final_box = [max(0, round(b)) for b in box]
            img_object = image.crop(final_box)
            image_objects.append((img_object, final_box))
        return image_objects

    def plot_results(self, pil_img, prob, boxes):
        img = pil_img.copy()
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            draw = D.Draw(img)
            draw.rectangle([(xmin, ymin), (xmax, ymax)])
        img.show()


##########################################################################################
##########################################################################################


class DETRObjectDetection:
    def __init__(self, show):
        self.show = show
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def detect_all_objects(self, image):
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
            outputs, target_sizes=target_sizes, threshold=0.9
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
        for box, _, _ in boxes:
            final_box = [max(0, round(b)) for b in box]
            img_object = image.crop(final_box)
            image_objects.append((img_object, final_box))
        return image_objects

    def plot_results(self, pil_img, boxes):
        img = pil_img.copy()
        for box, _, _ in boxes:
            xmin, ymin, xmax, ymax = box
            draw = D.Draw(img)
            draw.rectangle([(xmin, ymin), (xmax, ymax)])
        img.show()
