import cv2
import numpy as np
import onnxruntime
import time
import os
import pickle
from tqdm import tqdm 
from itertools import compress
from deel.puncc.api.utils import hungarian_assignment
from utils.helpers import nms, xywh2xyxy, draw_detections

class YOLOAPIWrappper:
    def __init__(self, path,file_path="calibration_results.pickle", conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.file_path = file_path
        # Initialize onnx model for inference
        self.initialize_model(path)

    def __call__(self, image):
        """
        Perform object detection on the input image.
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Detected bounding boxes, scores, and class IDs.
        """
        return self.detect_objects(image)

    def initialize_model(self, path):
        """
        Initialize the ONNX model for inference.
        Args:
            path (str): Path to the ONNX model file.
        """
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        """
        Perform object detection on the input image.
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Detected bounding boxes, scores, and class IDs.
        """
        input_tensor = self.prepare_input(image)
        output = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(output)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        """
        Prepare the input image for the model. 
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            np.ndarray: Preprocessed input tensor.
        """
        self.img_height, self.img_width = image.shape[:2] # Get original image dimensions
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        input_img = cv2.resize(input_img, (self.input_width, self.input_height)) # Resize input image

        input_img = input_img / 255.0 # Normalize pixel values
        input_img = input_img.transpose(2, 0, 1) # Change to (C, H, W) cause torch default to channel last
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32) # Add batch dimension & convert to float32 as opencv default's different

        return input_tensor 

    def inference(self, input_tensor):
        """
        Perform inference on the input tensor using the ONNX model.
        Args:
            input_tensor (np.ndarray): Preprocessed input tensor.
        Returns:
            np.ndarray: Model output.
        """
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        return outputs

    def process_output(self, output):
        """
        Process the model output to extract bounding boxes, scores, and class IDs.
        Args:
            output (np.ndarray): Model output.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Detected bounding boxes, scores, and class IDs.
        """
        predictions = np.squeeze(output) # Get the output predictions without batch dimension

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4] # Get object confidence scores
        predictions = predictions[obj_conf > self.conf_threshold] # Filter predictions based on confidence threshold
        obj_conf = obj_conf[obj_conf > self.conf_threshold] # Get the filtered object confidence scores

        predictions[:, 5:] *= obj_conf[:, np.newaxis] # Multiply class confidence with object confidence to get final confidence scores

        # Get the scores and filter predictions
        scores = np.max(predictions[:, 5:], axis=1) # Get the maximum class confidence scores
        predictions = predictions[scores > self.conf_threshold] # Filter predictions based on final confidence scores
        scores = scores[scores > self.conf_threshold] # Get the filtered scores

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1) # Get the class IDs

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        """
        Extract and scale bounding boxes from the model predictions.
        Args:
            predictions (np.ndarray): Model predictions.
        Returns:
            np.ndarray: Scaled bounding boxes in (x1, y1, x2, y2) format.
        """
        # Extract and scale boxes
        boxes = predictions[:, :4]
        boxes /= np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        boxes = xywh2xyxy(boxes)
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4, save_path=None):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha, save_path)

    def get_input_details(self):
        """
        Get input details of the ONNX model.
        """
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        """
        Get output details of the ONNX model.
        """
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def predict_from_image(self, image_path):
        """
        Predict bounding boxes for a single image.
        Args:
            image_path (str): Path to the input image.
        Returns:
            np.ndarray: Predicted bounding boxes in (x1, y1, x2, y2) format.
            np.ndarray: Scores for each bounding box.
        """
        image = cv2.imread(image_path)
        self.detect_objects(image)
        return self.boxes, self.scores

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
            y_preds_per_image, scores_per_image = self.predict_from_image(image_path)
            
            #y_preds_per_image = np.array(y_preds_per_image)
            #y_trues_per_image = np.array(y_trues_per_image)
            if y_preds_per_image is None or y_preds_per_image.shape[0] == 0:
                return np.zeros((0, 4)), np.zeros((0, 4)), np.array([], dtype=bool), np.array([], dtype=float) 
            
            y_preds_per_image = np.array(y_preds_per_image, dtype=np.float64)
            y_trues_per_image = np.array(y_trues_per_image, dtype=np.float64)
            scores_per_image = np.array(scores_per_image, dtype=np.float64)

            print("Predicted boxes shape:", np.array(y_preds_per_image).shape)
            print("True boxes shape:", np.array(y_trues_per_image).shape)
            print(f"y_preds_per_image: {y_preds_per_image}, dtype: {y_preds_per_image.dtype}")
            print(f"y_trues_per_image: {y_trues_per_image}, dtype: {y_trues_per_image.dtype}")
            print(f"scores_per_image: {scores_per_image}, dtype: {scores_per_image.dtype}")

            y_preds_i, y_trues_i, indices_i = hungarian_assignment(
            np.array(y_preds_per_image) , np.array(y_trues_per_image).astype(float), min_iou=min_iou
            )

            return y_preds_i, y_trues_i, indices_i, scores_per_image

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
                    results_dict["scores"] ,
                )
        else:
            raise FileNotFoundError(f"No results file found at {self.file_path}.")

    def save_results(self, y_preds, y_trues, images, labels, scores):
        """
        Save results to a file.
        Args:
            y_preds (list): Predicted bounding boxes.
            y_trues (list): True bounding boxes.
            images (list): Image paths.
            labels (list): Associated labels or metadata.
            matched_scores (list): Matched scores for the predictions.
        """
        with open(self.file_path, "wb") as file:
            pickle.dump(
                {"y_preds": y_preds, "y_trues": y_trues, "images": images, "labels": labels, "scores": scores}, file
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
        y_preds, matched_trues, images, classes, scores = [], [], [], [], []

        # Check if results already exist
        if os.path.exists(self.file_path):
            return self.load_results()

        # Iterate over the dataset
        for counter, (image_path, y_true, label) in enumerate(
            tqdm(zip(image_paths, y_trues, labels), total=len(image_paths))
        ):
            # Predict and match bounding boxes
            y_preds_i, y_trues_i, indices_i, scores_i = self.predict_and_match(image_path, y_true, min_iou=min_iou)
            scores.append(scores_i)
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
        scores = np.concatenate(scores, axis=0)
        
        self.save_results(y_preds, matched_trues, images, classes, scores)

        return y_preds, matched_trues, images, classes, scores
    
    

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

    