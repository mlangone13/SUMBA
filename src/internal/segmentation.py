import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from PIL import Image
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

from ultralytics import YOLO
import ultralytics

##########################################################################################
##########################################################################################


class YoloV8Segmentation:
    def __init__(self, min_mask_size, show):
        print("[SEGMENTATION] - Using Yolo V8 Segmentation Backend")
        ultralytics.checks()
        self.show = show
        self.model = YOLO("yolov8x-seg.pt")
        self.min_mask_size = min_mask_size

    def segment_object(self, first_object):

        results = self.model.predict(source=first_object, conf=0.1)

        if len(results[0]) == 0:
            print("No segmented object found")
            raise (Exception("No segmented object found"))

        # Step: Select object value for segmentation
        label_id, label = self.get_master_object_id(results)

        if self.show:
            output_array = results[0].plot()
            print(Image.fromarray(output_array))

        src_gray = self.process_mask(results, label_id)

        return src_gray, label

    def get_master_object_id(self, results):

        height, width, _ = results[0].orig_img.shape
        image_size = height * width

        for id, segmented_object in enumerate(results[0]):
            segment_object_size = np.sum(segmented_object.masks.data.numpy())

            if segment_object_size > self.min_mask_size * image_size:
                element = segmented_object.boxes
                cls = int(element.cls[0])
                name = segmented_object.names[cls]
                return id, name

        segmented_object = results[0][0]
        element = segmented_object.boxes
        cls = int(element.cls[0])
        name = segmented_object.names[cls]
        return 0, name

    def process_mask(self, results, label_id):
        src_seg = np.array(results[0].masks.data[label_id])
        src_gray = np.uint8(src_seg)
        src_gray[src_gray == 1] = 255
        return src_gray


class MaskFormerSegmentation:
    def __init__(self, min_mask_size, show):
        print("[SEGMENTATION] - Using MaskFormer Segmentation Backend")
        self.show = show
        # load MaskFormer fine-tuned on COCO panoptic segmentation
        self.feature_extractor = MaskFormerImageProcessor.from_pretrained(
            "facebook/maskformer-swin-tiny-coco"
        )
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-tiny-coco"
        )
        self.min_mask_size = min_mask_size

    def segment_object(self, first_object):
        # Step 3: Generate segmentation mask
        outputs = self.generate_mask_on_object(first_object)
        semantic_segmentation = (
            self.feature_extractor.post_process_semantic_segmentation(outputs)[0]
        )

        # Step 4 : Select object value for segmentation
        label_id, label = self.get_master_object_id(semantic_segmentation)

        if self.show:
            self.draw_semantic_segmentation(semantic_segmentation)
            print(f"label: {label} - label_id: {label_id}")

        src_gray = self.process_mask(semantic_segmentation, label_id)
        return src_gray, label

    def get_master_object_id(self, segmentation):
        labels_ids = torch.unique(segmentation).tolist()
        unique, counts = np.unique(segmentation.numpy(), return_counts=True)
        dict_count = dict(zip(unique, counts))
        total_pixels = np.sum([dict_count[k] for k in dict_count.keys()])
        for id in labels_ids:
            if dict_count[id] > total_pixels * self.min_mask_size:
                return id, self.model.config.id2label[id]

        return labels_ids[0], self.model.config.id2label[labels_ids[0]]

    def generate_mask_on_object(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        return outputs

    def process_mask(self, semantic_segmentation, label_id):
        src_seg = semantic_segmentation.numpy().copy()
        src_gray = np.uint8(src_seg)
        src_gray[src_gray != label_id] = 0
        src_gray[src_gray == label_id] = 255
        return src_gray

    def draw_semantic_segmentation(self, segmentation):
        # get the used color map
        viridis = cm.get_cmap("viridis", torch.max(segmentation))
        # get all the unique numbers
        labels_ids = torch.unique(segmentation).tolist()
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        handles = []
        for label_id in labels_ids:
            label = self.model.config.id2label[label_id]
            color = viridis(label_id)
            handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=handles)
        return fig

    def draw_panoptic_segmentation(self, segmentation, segments_info):
        # get the used color map
        viridis = cm.get_cmap("viridis", torch.max(segmentation))
        fig, ax = plt.subplots()
        ax.imshow(segmentation)
        instances_counter = defaultdict(int)
        handles = []
        # for each segment, draw its legend
        for segment in segments_info:
            segment_id = segment["id"]
            print(segment)
            segment_label_id = segment["label_id"]
            segment_label = self.model.config.id2label[segment_label_id]
            label = f"{segment_label}-{instances_counter[segment_label_id]}"
            instances_counter[segment_label_id] += 1
            color = viridis(segment_id)
            handles.append(mpatches.Patch(color=color, label=label))

        ax.legend(handles=handles)
        return fig
