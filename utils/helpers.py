import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# UTILS FUNCTIONS
def nms(boxes, scores, iou_threshold):
    """
    Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
    Args:
        boxes (np.ndarray): Array of bounding boxes with shape (N, 4).
        scores (np.ndarray): Array of scores for each box with shape (N,).
        iou_threshold (float): IoU threshold for suppression.
    Returns:
        keep_boxes (list): List of indices of boxes to keep.
    """
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

def compute_iou(box, boxes):
    """
    Compute Intersection over Union (IoU) between a box and an array of boxes.
    Args:
        box (np.ndarray): A single bounding box with shape (4,).
        boxes (np.ndarray): Array of bounding boxes with shape (N, 4).
    Returns:
        iou (np.ndarray): Array of IoU values with shape (N,).
    """
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

def iou(box_a, box_b): 
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Args:
        box_a (np.ndarray): First bounding box with shape (4,).
        box_b (np.ndarray): Second bounding box with shape (4,).
    Returns:
        iou (float): IoU value.
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = interArea / max( 1e-6, float(boxAArea + boxBArea - interArea ))
    return iou

def ioa(box_a, box_b): 
    """
    Compute Intersection over Area (IoA) between two bounding boxes.
    Args:
        box_a (np.ndarray): First bounding box with shape (4,).
        box_b (np.ndarray): Second bounding box with shape (4,).
    Returns:
        ioa (float): IoA value.
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    ioa = interArea / float(boxBArea + 1e-6)
    return ioa
  

def xywh2xyxy(x):
    """
    Convert bounding boxes from (x, y, width, height) format to (x1, y1, x2, y2) format.
    Args:
        x (np.ndarray): Array of bounding boxes with shape (N, 4) in (x, y, width, height) format.
    Returns:    
        y (np.ndarray): Array of bounding boxes with shape (N, 4) in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, save_path=None):
    """
    Draw bounding boxes and labels on the image.
    args:
        image (np.ndarray): Input image.
        boxes (np.ndarray): Array of bounding boxes with shape (N, 4).
        scores (np.ndarray): Array of scores for each box with shape (N,).
        class_ids (np.ndarray): Array of class IDs for each box with shape (N,).
        save_path (str): Path to save the output image.
    Returns:
        None
    """
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