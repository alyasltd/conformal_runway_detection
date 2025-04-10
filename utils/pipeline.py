import os
import numpy as np
from wrapper import YOLOAPIWrappper
import glob
import pandas as pd 
from PIL import Image
from deel.puncc.api.prediction import IdPredictor
from deel.puncc.object_detection import SplitBoxWise
from sklearn.model_selection import train_test_split
from deel.puncc.plotting import draw_bounding_box
from deel.puncc.metrics import object_detection_mean_coverage, object_detection_mean_area

 
class CPPipeline:
    def __init__(self, yolo_wrapper, test_set='test', method="multiplicative"):
        """
        Initialise avec une instance de YOLOAPIWrapper et le jeu de test à utiliser.
        """
        self.method = method
        self.yolo_wrapper = yolo_wrapper  
        self.dataset_path = "/home/aws_install/data/yolo_database"

        # Permet de choisir test, test_synth, ou test_real
        self.images_path = os.path.join(self.dataset_path, f"images/{test_set}")
        self.labels_path = os.path.join(self.dataset_path, f"labels/{test_set}")

        #self.metadata_path = "/home/aws_install/conformal_prediction/output_with_range.csv"

    def parse_data_distance(self, aire_min, aire_max):
        """
        Filtre les données selon l'intervalle de bbox_area et charge les images & labels correspondants.

        Args:
            aire_min (int): Valeur minimale de bbox_area.
            aire_max (int): Valeur maximale de bbox_area.
            test_size (float): Proportion des données à utiliser pour le test.
            random_state (int): Seed pour reproductibilité.

        Returns:
            Tuple[List[str], List[str]]: Listes des chemins d'images et labels filtrés.
        """
        # Charger les métadonnées
        df = pd.read_csv(self.metadata_path)

        # Filtrer selon bbox_area
        df_filtered = df[(df["bbox_area"] >= aire_min) & (df["bbox_area"] <= aire_max)]
        
        if df_filtered.empty:
            print(f"Aucune donnée trouvée pour l'intervalle bbox_area [{aire_min}, {aire_max}].")
            return [], []
        
        # Récupérer les chemins des images et labels
        image_paths = [os.path.join(self.val_images_path, img) for img in df_filtered["image"]]
        label_paths = [os.path.join(self.val_labels_path, img.replace(".jpeg", ".txt").replace(".png", ".txt")) 
                       for img in df_filtered["image"]]

        # Vérifier que les fichiers existent
        image_paths = [img for img in image_paths if os.path.exists(img)]
        label_paths = [lbl for lbl in label_paths if os.path.exists(lbl)]

        print(f"Nombre d'images retenues : {len(image_paths)}")
        return image_paths, label_paths
    
    def extract_yolo_dataset(self, random_seed=42):
        """
        Extracts image paths, ground truth boxes, and labels from a list of image and label paths.

        Args:
            image_paths (List[str]): List of image file paths.
            label_paths (List[str]): List of label file paths.
            random_seed (int): Random seed for reproducibility.

        Returns:
            Tuple[List[str], List[np.ndarray], List[List[int]]]: Image paths, ground truth boxes, and labels.
        """
        np.random.seed(random_seed)
        label_files = sorted(glob.glob(os.path.join(self.labels_path, "*.txt")))  # Get all .txt label files
        image_files = sorted(glob.glob(os.path.join(self.images_path, "*.jpg")) + 
                     glob.glob(os.path.join(self.images_path, "*.jpeg")) + 
                     glob.glob(os.path.join(self.images_path, "*.png")))


        ground_truth_boxes = []
        all_labels = []
        valid_image_paths = []  # Store valid images (some may be missing)

        for label_file, image_file in zip(label_files, image_files):  
            #print(label_file, image_file)

            if not os.path.exists(label_file):
                print(f"Warning: Label file not found {label_file}. Skipping.")
                continue

            if not os.path.exists(image_file):
                print(f"Warning: Image file not found {image_file}. Skipping.")
                continue

            with open(label_file, "r") as f:
                labels = f.readlines()

            image = Image.open(image_file)
            image_width, image_height = image.size

            boxes = []
            labels_per_image = []

            for label in labels:
                parts = list(map(float, label.strip().split()))
                if len(parts) != 5:
                    print(f"Warning: Incorrect label format in {label_file}. Skipping.")
                    continue

                class_id, x_center, y_center, width, height = parts
                x_center *= image_width
                y_center *= image_height
                width *= image_width
                height *= image_height
                x1 = x_center - (width / 2)
                y1 = y_center - (height / 2)
                x2 = x_center + (width / 2)
                y2 = y_center + (height / 2)
                boxes.append([x1, y1, x2, y2])
                labels_per_image.append(int(class_id))

            valid_image_paths.append(image_file)
            ground_truth_boxes.append(np.array(boxes))
            all_labels.append(labels_per_image)

        return np.array(valid_image_paths), np.array(ground_truth_boxes, dtype=object), np.array(all_labels, dtype=object)

    def calibration_and_val(self, X, y, labels) : 
        X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(X, y, labels, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val, labels_train, labels_val
    
    def pipeline(self, X_train, y_train, labels_train):
        api_model = IdPredictor()
        # we call the query method to get the predictions and the matched ground truth boxes which are our calibration data
        y_preds, y_trues_matched, images, classes = self.yolo_wrapper.query(X_train, y_train, labels_train)

        print("Predictions:", y_preds)
        print("Matched Ground Truths:", y_trues_matched)
        print("Images:", images)
        print("Classes:", classes)

        # we instantiate the conformal predictor
        conformal_predictor = SplitBoxWise(api_model, method=self.method, train=False)

        # fitting the conformal predictor on the calibration data
        conformal_predictor.fit(X_calib=y_preds, y_calib=y_trues_matched)
        return conformal_predictor

    def infer_eval_single_image(self, conformal_predictor, image_path, bboxes, classes, y_new_api): 
        # Predict only if not provided (for standalone use)
        if y_new_api is None:
            y_new_api = self.yolo_wrapper.predict_from_image(image_path)

        # Coverage target
        alpha = 0.3

        # Inference + Uncertainty Quantification
        y_pred_new, box_inner, box_outer = conformal_predictor.predict(y_new_api, alpha=alpha)

        # Convert data for visualization
        classes = [str(class_) for class_ in classes]
        y_pred_new_t = tuple(map(tuple, y_pred_new))
        bboxes_t = tuple(map(tuple, bboxes))
        box_outer_t = tuple(map(tuple, box_outer))
        image = Image.open(image_path)

        # Draw bounding boxes
        for i in range(len(y_pred_new)):
            image_with_bbox = draw_bounding_box(
                image=image,
                box=bboxes_t[i],
                label=classes[i],
                legend="Truth",
                color="red",
            )
            image_with_bbox = draw_bounding_box(
                image=image,
                box=y_pred_new_t[i],
                legend="Predictions",
                color="blue",
            )
            image_with_bbox = draw_bounding_box(
                image=image,
                box=box_outer_t[i],
                legend="Conformalized Outer Box",
                color="orange",
            )
            
            # image_with_bbox = draw_bounding_box(
            #     image=image_with_bbox,
            #     box=box_inner[i],
            #     legend="Conformalized Inner Box",
            #     color="brown",
            # )

        _ = draw_bounding_box(image=image, show=True)

        # Compute and print metrics
        coverage = object_detection_mean_coverage(box_outer, bboxes)
        average_area = object_detection_mean_area(box_outer)
        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average area: {np.round(average_area, 2)}")


    def infer_eval_all(self, X_val, y_val, labels_val, conformal_predictor, visualize=False):
        no_predictions = 0
        y_pred_val, y_true_val, images_val, classes_val, box_inner_val, box_outer_val = [], [], [], [], [], []

        for i in range(len(X_val)):
            image_path, y_true, classes = X_val[i], y_val[i], labels_val[i]
            y_new_api = self.yolo_wrapper.predict_from_image(image_path)

            image = Image.open(image_path)

            if y_new_api.shape[0] == 0:  
                print(f"No detections in image {image_path}")
                no_predictions += 1
                continue  

            alpha = 0.3
            y_pred_new, box_inner, box_outer = conformal_predictor.predict(y_new_api, alpha=alpha)

            # Sorting for consistency
            sort_indices_outer = np.lexsort(box_outer.T[::-1])
            box_outer = box_outer[sort_indices_outer]
            sort_indices_true = np.lexsort(y_true.T[::-1])
            y_true = y_true[sort_indices_true]

            # Append results
            y_pred_val.append(y_pred_new)
            y_true_val.append(y_true)
            images_val.append(image)
            classes_val.append(classes)
            box_inner_val.append(box_inner)
            box_outer_val.append(box_outer)

            # Visualize the first 5 images for debugging
            if visualize and i < 5:
                self.infer_eval_single_image(conformal_predictor, image_path, y_true, classes, y_new_api)

        # Compute global metrics
        average_glo_area = [object_detection_mean_area(box_outer_val[i]) for i in range(len(box_outer_val)) if len(box_outer_val[i]) > 0]
        print(f"Average area of prediction intervals: {np.mean(average_glo_area)}")
        print(f"Average length of prediction intervals: {np.sqrt(np.mean(average_glo_area))}")
        print(f"Number of images without predictions: {no_predictions}")
        print(f"Number of images with predictions: {len(X_val) - no_predictions}")
        print(f"Number of images: {len(X_val)}")

        cover = [object_detection_mean_coverage(box_outer_val[i], y_true_val[i]) 
                for i in range(len(y_pred_val)) if box_outer_val[i].shape == y_true_val[i].shape]
        print(f"Marginal coverage: {np.mean(cover)}")

        return y_pred_val, y_true_val, box_outer_val, images_val # pred_yolo, gt, pred_cp
    

    #TODO 
    #function average map50 map5095 over all images
    #function - Iou(pred_c, gt) given IoU(pred, gt) - IoA(pred_c, gt) given IoA(pred, gt)
    #-  IoA(pred_c, gt),  IoU(pred_c, gt) - IoU(pred_c, gt), slant distance

