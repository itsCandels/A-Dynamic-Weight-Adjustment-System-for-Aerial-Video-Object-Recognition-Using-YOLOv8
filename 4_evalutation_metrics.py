'''
import cv2
import numpy as np
import datetime
import time
import threading
import os
from queue import Queue, Full, Empty
from ultralytics import YOLO
import gc
import torch
from collections import deque

class BasicMotionDetector:
    def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=500):
        self.accumWeight = accumWeight
        self.deltaThresh = deltaThresh
        self.minArea = minArea
        self.avg = None

    def update(self, image):
        locs = []
        if self.avg is None:
            self.avg = image.astype("float")
            return locs

        cv2.accumulateWeighted(image, self.avg, self.accumWeight)
        frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))
        thresh = cv2.threshold(frameDelta, self.deltaThresh, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            if cv2.contourArea(c) < self.minArea:
                continue
            locs.append(c)

        return locs

def simulate_image_intensifier(image, amplification_factor, noise_level=0.05):
    image_float = image.astype(np.float32) / 255.0
    amplified_image = image_float * amplification_factor
    noise = np.random.normal(scale=noise_level, size=image.shape)
    noisy_image = amplified_image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    phosphor_image = (noisy_image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(phosphor_image)
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel)
    return sharpened_image

def calculate_brightness(image):
    return np.mean(image)

def apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = calculate_brightness(gray_frame)
    if brightness < brightness_threshold:
        intensified_frame = simulate_image_intensifier(gray_frame, amplification_factor, noise_level)
        intensified_frame = cv2.cvtColor(intensified_frame, cv2.COLOR_GRAY2BGR)
    else:
        intensified_frame = frame
    return intensified_frame

def resize_and_pad_image(image, target_size):
    ih, iw = image.shape[:2]
    h, w = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image_resized = cv2.resize(image, (nw, nh))

    top, left = (h - nh) // 2, (w - nw) // 2
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    new_image[top:top + nh, left:left + nw] = image_resized

    return new_image, scale, left, top

def read_frames(cap, frame_queue, skip_frames=1, target_size=(640, 480)):
    frame_id = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put((None, None))
                break

            frame = cv2.resize(frame, target_size)
            if frame_id % skip_frames == 0:
                frame_queue.put((frame, frame_id))
            frame_id += 1
        except Full:
            time.sleep(0.1)

def save_annotation(frame_id, boxes, annotations_dir, scale, left, top, original_size):
    with open(os.path.join(annotations_dir, f"{frame_id:06d}.txt"), 'w') as f:
        for box in boxes:
            cls_id, x1, y1, x2, y2 = box
            # Convert the coordinates back to the original image size
            x1 = (x1 - left) / scale
            y1 = (y1 - top) / scale
            x2 = (x2 - left) / scale
            y2 = (y2 - top) / scale
            # Normalize the coordinates
            x_center = (x1 + x2) / 2 / original_size[1]
            y_center = (y1 + y2) / 2 / original_size[0]
            width = (x2 - x1) / original_size[1]
            height = (y2 - y1) / original_size[0]
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

def save_image(frame, frame_id, images_dir):
    cv2.imwrite(os.path.join(images_dir, f"{frame_id:06d}.jpg"), frame)

def process_frames(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold, model_yolo, model_visdrone, device, annotations_dir, images_dir):
    target_size = (640, 640)
    frame_id = 0
    current_model = model_yolo
    initial_box_size = None
    box_size_history = deque(maxlen=30)

    while True:
        try:
            frame_data = frame_queue.get(timeout=1)
            if frame_data is None:
                processed_queue.put((None, None))
                break

            frame, frame_id = frame_data
            if frame is None:
                continue

            original_height, original_width = frame.shape[:2]
            night_vision_frame = apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold)
            resized_frame, scale, left, top = resize_and_pad_image(night_vision_frame, target_size)

            resized_frame_tensor = torch.from_numpy(resized_frame).to(device).float() / 255.0
            resized_frame_tensor = resized_frame_tensor.permute(2, 0, 1).unsqueeze(0)

            results = current_model(resized_frame_tensor, conf=0.25, iou=0.25)[0]

            box_sizes = []
            boxes = []

            for box in results.boxes:
                cls_id = int(box.cls)
                if cls_id in current_model.names:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = int((x1 - left) / scale)
                    y1 = int((y1 - top) / scale)
                    x2 = int((x2 - left) / scale)
                    y2 = int((y2 - top) / scale)

                    box_size = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    box_sizes.append(box_size)
                    box_size_history.append(box_size)

                    if initial_box_size is None:
                        initial_box_size = box_size
                        print(f"Initial box size: {initial_box_size}")

                    cv2.rectangle(night_vision_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(night_vision_frame, current_model.names[cls_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    boxes.append((cls_id, x1, y1, x2, y2))

            if len(box_size_history) == 30:
                moving_average_box_size = np.mean(box_size_history)
                print(f"Moving average box size: {moving_average_box_size}")

                if moving_average_box_size < initial_box_size * 3 / 4:
                    if current_model != model_visdrone:
                        current_model = model_visdrone
                        print("Switching to VisDrone model due to reduction in box size.")
                elif moving_average_box_size >= initial_box_size:
                    if current_model != model_yolo:
                        current_model = model_yolo
                        print("Switching to YOLO model due to increase in box size.")

            timestamp = datetime.datetime.now()
            ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(night_vision_frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            processed_queue.put((night_vision_frame, None))

            save_annotation(frame_id, boxes, annotations_dir, scale, left, top, (original_height, original_width))
            save_image(night_vision_frame, frame_id, images_dir)

            frame_id += 1
        except Empty:
            continue
        finally:
            gc.collect()

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA backend (NVIDIA GPU)")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS backend (Apple Silicon M1/M2)")
        return torch.device('mps')
    else:
        print("Using CPU backend")
        return torch.device('cpu')

def main():
    use_streaming = False
    video_path = "/Users/federicocandela/Desktop/TEST_ALGORITMO/5)_VIDEO_NOTTURNA_1_PASSEGGIATA/DJI_0590.MP4"

    if use_streaming:
        rtmp_url = "rtmp://localhost:1935/live"
        print(f"Tentativo di connessione al flusso RTMP: {rtmp_url}")

        cap = cv2.VideoCapture(rtmp_url)
        start_time = time.time()
        timeout = 10

        while not cap.isOpened():
            if time.time() - start_time > timeout:
                print("Errore: Timeout durante la connessione allo streaming video.")
                use_streaming = False
                break
            print("Tentativo di connessione in corso...")
            time.sleep(1)
            cap.open(rtmp_url)

        if use_streaming:
            print("Streaming video aperto con successo. Inizio della riproduzione...")

    if not use_streaming:
        print(f"Apertura del video locale: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {video_path}")
            return

    model_yolo = YOLO("/Users/federicocandela/Desktop/YOLO_OBJECT_DETECTION/1_RILEVATORE_MOVIMENTO_VIDEO/yolo_weights/yolov8n.pt")
    model_visdrone = YOLO("/Users/federicocandela/Desktop/YOLO_OBJECT_DETECTION/1_RILEVATORE_MOVIMENTO_VIDEO/yolo_weights/VisDrone_best.pt")

    amplification_factor = 10.0
    noise_level = 0.05
    brightness_threshold = 80
    skip_frames = 1

    frame_queue = Queue(maxsize=100)
    processed_queue = Queue(maxsize=100)

    device = get_device()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/5)_VIDEO_NOTTURNA_1_PASSEGGIATA/DYNAMIC/annotations'
    images_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/5)_VIDEO_NOTTURNA_1_PASSEGGIATA/DYNAMIC/images'

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, skip_frames, (640, 480)))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold, model_yolo, model_visdrone, device, annotations_dir, images_dir))

    read_thread.start()
    process_thread.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('REAL_TIME_VIDEO.mp4', fourcc, fps, (640, 480))

    while True:
        try:
            frame_data = processed_queue.get(timeout=1)
            if frame_data is None:
                break
            frame, _ = frame_data
            if frame is not None and frame.size > 0:
                out.write(frame)
                cv2.imshow('Night Vision', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Empty:
            continue

    read_thread.join()
    process_thread.join()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
'''
'''
import os
import csv

# Mappa delle classi tra VisDrone e YOLOv8 (aggiungi altre se necessario)
class_map = {
    0: 0,  # 'person'/'pedestrian' in YOLOv8 e VisDrone
    2: 3,  # 'car' in YOLOv8 diventa 3 in VisDrone
}

def read_annotations(file_path):
    with open(file_path, 'r') as file:
        annotations = [line.strip().split() for line in file.readlines()]
    return [[int(a[0])] + list(map(float, a[1:])) for a in annotations]

def convert_yolo_to_bbox(yolo_annotation, img_width, img_height, class_map=None):
    class_id, x_center, y_center, width, height = yolo_annotation
    if class_map:
        class_id = class_map.get(class_id, class_id)
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return (class_id, x1, y1, x2, y2)

def iou(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[3] - boxA[1]) * (boxA[4] - boxA[2])
    boxBArea = (boxB[3] - boxB[1]) * (boxB[4] - boxB[2])
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_metrics(preds, targets, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    target_used = [False] * len(targets)
    
    for pred in preds:
        match_found = False
        for i, target in enumerate(targets):
            if not target_used[i] and pred[0] == target[0] and iou(pred, target) > iou_threshold:
                true_positives += 1
                match_found = True
                target_used[i] = True
                break
        if not match_found:
            false_positives += 1
    
    false_negatives = len(targets) - sum(target_used)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall, false_positives, false_negatives, true_positives

def create_majority_consensus(yolo_annotations, visdrone_annotations, dynamic_annotations, iou_threshold=0.5):
    consensus_annotations = []
    all_annotations = yolo_annotations + visdrone_annotations + dynamic_annotations

    used_annotations = set()

    for ann in all_annotations:
        ann_tuple = tuple(ann)
        if ann_tuple in used_annotations:
            continue
        
        similar_annotations = [
            tuple(a) for a in all_annotations 
            if a[0] == ann[0] and iou(a, ann) > iou_threshold and tuple(a) not in used_annotations
        ]
        
        if len(similar_annotations) >= 2:  # Richiede che almeno due annotazioni siano simili
            consensus_annotations.append(ann_tuple)
            used_annotations.update(similar_annotations)
    
    return consensus_annotations

def main():
    dynamic_annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/1)_VIDEO_1_VIA_MARINA_SPIAGGIA/DYNAMIC/annotations'
    yolo_annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/1)_VIDEO_1_VIA_MARINA_SPIAGGIA/VIS_DRONE/annotations'
    visdrone_annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/1)_VIDEO_1_VIA_MARINA_SPIAGGIA/YOLOV8/annotations'

    img_width = 640
    img_height = 480

    annotation_files = [f.split('.')[0] for f in os.listdir(dynamic_annotations_dir) if f.endswith('.txt')]

    for file in annotation_files:
        yolo_annotation_file = os.path.join(yolo_annotations_dir, f'{file}.txt')
        visdrone_annotation_file = os.path.join(visdrone_annotations_dir, f'{file}.txt')
        dynamic_annotation_file = os.path.join(dynamic_annotations_dir, f'{file}.txt')
        
        if os.path.exists(yolo_annotation_file) and os.path.exists(visdrone_annotation_file) and os.path.exists(dynamic_annotation_file):
            yolo_anns = read_annotations(yolo_annotation_file)
            visdrone_anns = read_annotations(visdrone_annotation_file)
            dynamic_anns = read_annotations(dynamic_annotation_file)
            
            yolo_bboxes = [convert_yolo_to_bbox(ann, img_width, img_height, class_map) for ann in yolo_anns]
            visdrone_bboxes = [convert_yolo_to_bbox(ann, img_width, img_height, class_map) for ann in visdrone_anns]
            dynamic_bboxes = [convert_yolo_to_bbox(ann, img_width, img_height) for ann in dynamic_anns]
            
            consensus_annotations = create_majority_consensus(yolo_bboxes, visdrone_bboxes, dynamic_bboxes)
            
            # Calcolo delle metriche rispetto a consensus_annotations
            precision_yolo, recall_yolo, _, _, _ = calculate_metrics(yolo_bboxes, consensus_annotations)
            precision_visdrone, recall_visdrone, _, _, _ = calculate_metrics(visdrone_bboxes, consensus_annotations)
            precision_dynamic, recall_dynamic, _, _, _ = calculate_metrics(dynamic_bboxes, consensus_annotations)

            print(f"File: {file}")
            print("YOLOv8 - Precision:", precision_yolo, "Recall:", recall_yolo)
            print("VisDrone - Precision:", precision_visdrone, "Recall:", recall_visdrone)
            print("Dynamic System - Precision:", precision_dynamic, "Recall:", recall_dynamic)

if __name__ == "__main__":
    main()

'''

import os
import csv
import matplotlib.pyplot as plt

# Mappa delle classi tra VisDrone e YOLOv8 (aggiungi altre se necessario)
class_map = {
    0: 0,  # 'person'/'pedestrian' in YOLOv8 e VisDrone
    2: 3,  # 'car' in YOLOv8 diventa 3 in VisDrone
}

def read_annotations(file_path):
    with open(file_path, 'r') as file:
        annotations = [line.strip().split() for line in file.readlines()]
    return [[int(a[0])] + list(map(float, a[1:])) for a in annotations]

def convert_yolo_to_bbox(yolo_annotation, img_width, img_height, class_map=None):
    class_id, x_center, y_center, width, height = yolo_annotation
    if class_map:
        class_id = class_map.get(class_id, class_id)
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return (class_id, x1, y1, x2, y2)

def iou(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[3] - boxA[1]) * (boxA[4] - boxA[2])
    boxBArea = (boxB[3] - boxB[1]) * (boxB[4] - boxB[2])
    return interArea / float(boxAArea + boxBArea - interArea)

def calculate_metrics(preds, targets, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    target_used = [False] * len(targets)
    
    for pred in preds:
        match_found = False
        for i, target in enumerate(targets):
            if not target_used[i] and pred[0] == target[0] and iou(pred, target) > iou_threshold:
                true_positives += 1
                match_found = True
                target_used[i] = True
                break
        if not match_found:
            false_positives += 1
    
    false_negatives = len(targets) - sum(target_used)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall, false_positives, false_negatives, true_positives

def create_majority_consensus(yolo_annotations, visdrone_annotations, dynamic_annotations, iou_threshold=0.5):
    consensus_annotations = []
    all_annotations = yolo_annotations + visdrone_annotations + dynamic_annotations

    used_annotations = set()

    for ann in all_annotations:
        ann_tuple = tuple(ann)
        if ann_tuple in used_annotations:
            continue
        
        similar_annotations = [
            tuple(a) for a in all_annotations 
            if a[0] == ann[0] and iou(a, ann) > iou_threshold and tuple(a) not in used_annotations
        ]
        
        if len(similar_annotations) >= 2:  # Richiede che almeno due annotazioni siano simili
            consensus_annotations.append(ann_tuple)
            used_annotations.update(similar_annotations)
    
    return consensus_annotations

def main():
    dynamic_annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/5)_VIDEO_NOTTURNA_1_PASSEGGIATA/DYNAMIC/annotations'
    yolo_annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/5)_VIDEO_NOTTURNA_1_PASSEGGIATA/VIS_DRONE/annotations'
    visdrone_annotations_dir = '/Users/federicocandela/Desktop/TEST_ALGORITMO/5)_VIDEO_NOTTURNA_1_PASSEGGIATA/YOLOV8/annotations'

    img_width = 640
    img_height = 480

    annotation_files = [f.split('.')[0] for f in os.listdir(dynamic_annotations_dir) if f.endswith('.txt')]

    total_precision_yolo = 0
    total_recall_yolo = 0
    total_precision_visdrone = 0
    total_recall_visdrone = 0
    total_precision_dynamic = 0
    total_recall_dynamic = 0
    num_files = 0

    for file in annotation_files:
        yolo_annotation_file = os.path.join(yolo_annotations_dir, f'{file}.txt')
        visdrone_annotation_file = os.path.join(visdrone_annotations_dir, f'{file}.txt')
        dynamic_annotation_file = os.path.join(dynamic_annotations_dir, f'{file}.txt')
        
        if os.path.exists(yolo_annotation_file) and os.path.exists(visdrone_annotation_file) and os.path.exists(dynamic_annotation_file):
            yolo_anns = read_annotations(yolo_annotation_file)
            visdrone_anns = read_annotations(visdrone_annotation_file)
            dynamic_anns = read_annotations(dynamic_annotation_file)
            
            yolo_bboxes = [convert_yolo_to_bbox(ann, img_width, img_height, class_map) for ann in yolo_anns]
            visdrone_bboxes = [convert_yolo_to_bbox(ann, img_width, img_height, class_map) for ann in visdrone_anns]
            dynamic_bboxes = [convert_yolo_to_bbox(ann, img_width, img_height) for ann in dynamic_anns]
            
            consensus_annotations = create_majority_consensus(yolo_bboxes, visdrone_bboxes, dynamic_bboxes)
            
            # Calcolo delle metriche rispetto a consensus_annotations
            precision_yolo, recall_yolo, _, _, _ = calculate_metrics(yolo_bboxes, consensus_annotations)
            precision_visdrone, recall_visdrone, _, _, _ = calculate_metrics(visdrone_bboxes, consensus_annotations)
            precision_dynamic, recall_dynamic, _, _, _ = calculate_metrics(dynamic_bboxes, consensus_annotations)

            total_precision_yolo += precision_yolo
            total_recall_yolo += recall_yolo
            total_precision_visdrone += precision_visdrone
            total_recall_visdrone += recall_visdrone
            total_precision_dynamic += precision_dynamic
            total_recall_dynamic += recall_dynamic
            num_files += 1

            print(f"File: {file}")
            print("YOLOv8 - Precision:", precision_yolo, "Recall:", recall_yolo)
            print("VisDrone - Precision:", precision_visdrone, "Recall:", recall_visdrone)
            print("Dynamic System - Precision:", precision_dynamic, "Recall:", recall_dynamic)

    if num_files > 0:
        avg_precision_yolo = total_precision_yolo / num_files
        avg_recall_yolo = total_recall_yolo / num_files
        avg_precision_visdrone = total_precision_visdrone / num_files
        avg_recall_visdrone = total_recall_visdrone / num_files
        avg_precision_dynamic = total_precision_dynamic / num_files
        avg_recall_dynamic = total_recall_dynamic / num_files

        print("\nAverage Metrics:")
        print("YOLOv8 - Average Precision:", avg_precision_yolo, "Average Recall:", avg_recall_yolo)
        print("VisDrone - Average Precision:", avg_precision_visdrone, "Average Recall:", avg_recall_visdrone)
        print("Dynamic System - Average Precision:", avg_precision_dynamic, "Average Recall:", avg_recall_dynamic)

        # Scrittura delle medie nel file CSV
        with open('average_results.csv', mode='w', newline='') as csv_file:
            fieldnames = ['System', 'Average Precision', 'Average Recall']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'System': 'YOLOv8', 'Average Precision': avg_precision_yolo, 'Average Recall': avg_recall_yolo})
            writer.writerow({'System': 'VisDrone', 'Average Precision': avg_precision_visdrone, 'Average Recall': avg_recall_visdrone})
            writer.writerow({'System': 'Dynamic System', 'Average Precision': avg_precision_dynamic, 'Average Recall': avg_recall_dynamic})

if __name__ == "__main__":
    main()