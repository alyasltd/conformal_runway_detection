import cv2
import numpy as np
import onnxruntime
import time
import os
import pickle
from tqdm import tqdm 
from itertools import compress
from deel.puncc.api.utils import hungarian_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLOAPIWrappper:
    def __init__(self, path,file_path="calibration_results.pickle", conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.file_path = file_path
        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        output = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(output)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Normalize pixel values
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output)

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores and filter predictions
        scores = np.max(predictions[:, 5:], axis=1)
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract and scale boxes
        boxes = predictions[:, :4]
        boxes /= np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        boxes = xywh2xyxy(boxes)
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4, save_path=None):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha, save_path)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def predict_from_image(self, image_path):
        """
        Predict bounding boxes for a single image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            np.ndarray: Predicted bounding boxes in (x1, y1, x2, y2) format.
        """
        image = cv2.imread(image_path)
        self.detect_objects(image)
        return self.boxes

    def predict_and_match(self, image_path, y_trues_per_image, min_iou=0.5):
            """
            Predict bounding boxes and match them with true bounding boxes using the Hungarian algorithm.
            Args:
                image_path (str): Path to the input image.
                y_trues_per_image (np.ndarray): True bounding boxes for the image.
                min_iou (float): Minimum IoU for valid matches but doesnt matter since we only have one boxe.
            Returns:
                Tuple[np.ndarray, np.ndarray]: Matched predicted and true bounding boxes.
            """
            y_preds_per_image = self.predict_from_image(image_path)
            
            #y_preds_per_image = np.array(y_preds_per_image)
            #y_trues_per_image = np.array(y_trues_per_image)
            if y_preds_per_image is None or y_preds_per_image.shape[0] == 0:
                return np.zeros((0, 4)), np.zeros((0, 4)), np.array([], dtype=bool)
            y_preds_per_image = np.array(y_preds_per_image, dtype=np.float64)
            y_trues_per_image = np.array(y_trues_per_image, dtype=np.float64)

            print("Predicted boxes shape:", np.array(y_preds_per_image).shape)
            print("True boxes shape:", np.array(y_trues_per_image).shape)
            print(f"y_preds_per_image: {y_preds_per_image}, dtype: {y_preds_per_image.dtype}")
            print(f"y_trues_per_image: {y_trues_per_image}, dtype: {y_trues_per_image.dtype}")

            y_preds_i, y_trues_i, indices_i = hungarian_assignment(
            np.array(y_preds_per_image) , np.array(y_trues_per_image).astype(float), min_iou=min_iou
            )
            return y_preds_i, y_trues_i, indices_i

    def load_results(self):
        """
        Load previously saved results from a file.
        Returns:
            Tuple: y_preds, y_trues, images, and labels.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as file:
                results_dict = pickle.load(file)
                return (
                    results_dict["y_preds"],
                    results_dict["y_trues"],
                    results_dict["images"],
                    results_dict["labels"],
                )
        else:
            raise FileNotFoundError(f"No results file found at {self.file_path}.")

    def save_results(self, y_preds, y_trues, images, labels):
        """
        Save results to a file.
        Args:
            y_preds (list): Predicted bounding boxes.
            y_trues (list): True bounding boxes.
            images (list): Image paths.
            labels (list): Associated labels or metadata.
        """
        with open(self.file_path, "wb") as file:
            pickle.dump(
                {"y_preds": y_preds, "y_trues": y_trues, "images": images, "labels": labels}, file
            )

    def query(self, image_paths, y_trues, labels, min_iou=0.5, n_instances=None):
        """
        Predict bounding boxes for a batch of images and match them to ground truth using Hungarian assignment.
        Args:
            image_paths (List[str]): List of image paths.
            y_trues (List[np.ndarray]): List of true bounding boxes for each image.
            labels (List[List[int]]): List of true labels for each bounding box.
            min_iou (float): Minimum IoU for valid matches.
            n_instances (int): Maximum number of images to process. If None, process all images.
        Returns:
            Tuple[np.ndarray, np.ndarray, list, np.ndarray]: Predictions, ground truths, image paths, and labels.
        """
        y_preds, matched_trues, images, classes = [], [], [], []

        # Check if results already exist
        if os.path.exists(self.file_path):
            return self.load_results()

        # Iterate over the dataset
        for counter, (image_path, y_true, label) in enumerate(
            tqdm(zip(image_paths, y_trues, labels), total=len(image_paths))
        ):
            # Predict and match bounding boxes
            y_preds_i, y_trues_i, indices_i = self.predict_and_match(image_path, y_true, min_iou=min_iou)
            y_preds.append(y_preds_i)
            matched_trues.append(y_trues_i)
            images.append(image_path)

            # Get matched classes
            classes.append(list(compress(label, indices_i)))

            # Stop if n_instances is reached
            if n_instances is not None and counter + 1 >= n_instances:
                break

        # Concatenate results
        y_preds = np.concatenate(y_preds, axis=0)
        matched_trues = np.concatenate(matched_trues, axis=0)
        classes = np.concatenate(classes, axis=0)

        # Save results
        self.save_results(y_preds, matched_trues, images, classes)

        return y_preds, matched_trues, images, classes



# UTILS FUNCTIONS
# NMS Helper Function
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

#Helper Function
def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    iou = intersection_area / union_area
    return iou

#TODO compute ioa
  
# Bounding box conversion helper function
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# Draw detections using matplotlib
def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, save_path=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    class_names = ["runway"]
    colors = ["red"]

    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        ax.text(x1, y1, caption, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)

    plt.show()

# TESTING
#if __name__ == '__main__':
    #model_path = "/home/aws_install/conformal_prediction/trainings/v6_small_pretrained/exp/weights/best_ckpt.onnx"
    #model_path = '/home/aws_install/yolov5/runs/train/lard_yolo_small_pretrained3/weights/best.onnx'
    #image_path = "/home/aws_install/data/yolo_database/images/test/DAAS_27_35_07.jpeg"


    #yolov6_detector = YOLOAPIWrappper(model_path, conf_thres=0.7, iou_thres=0.5)
    #image = cv2.imread(image_path)

    # Perform object detection
    #yolov6_detector(image)

    # Draw detections on the image
    #save_path = "output_image.png"  # Save image as output_image.png
    #yolov6_detector.draw_detections(image, save_path=save_path)

    