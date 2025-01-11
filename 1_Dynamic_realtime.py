import cv2
import numpy as np
import datetime
import time
import threading
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
        ret, frame = cap.read()
        if not ret:
            frame_queue.put((None, None))  # Segnale di terminazione
            break

        frame = cv2.resize(frame, target_size)
        if frame_id % skip_frames == 0:
            frame_queue.put((frame, None))
        frame_id += 1

def process_frames(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold, model_yolo, model_visdrone, device):
    target_size = (640, 640)
    frame_id = 0
    current_model = model_yolo
    initial_box_size = None
    box_size_history = deque(maxlen=30)

    while True:
        try:
            frame, _ = frame_queue.get(timeout=1)
            if frame is None:
                processed_queue.put((None, None))  # Segnale di terminazione
                break

            original_height, original_width = frame.shape[:2]
            night_vision_frame = apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold)
            resized_frame, scale, left, top = resize_and_pad_image(night_vision_frame, target_size)

            resized_frame_tensor = torch.from_numpy(resized_frame).to(device).float() / 255.0
            resized_frame_tensor = resized_frame_tensor.permute(2, 0, 1).unsqueeze(0)

            results = current_model(resized_frame_tensor)[0]

            box_sizes = []
            for box in results.boxes:
                cls_id = int(box.cls)
                if cls_id in current_model.names:
                    cls_name = current_model.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x1 = int((x1 - left) / scale)
                    y1 = int((y1 - top) / scale)
                    x2 = int((x2 - left) / scale)
                    y2 = int((y2 - top) / scale)

                    box_size = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    box_sizes.append(box_size)

                    cv2.rectangle(night_vision_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(night_vision_frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Modifiche apportate qui
            if box_sizes:
                average_box_size = np.mean(box_sizes)
                box_size_history.append(average_box_size)
                print(f"Average box size for this frame: {average_box_size}")

                if initial_box_size is None:
                    initial_box_size = average_box_size
                    print(f"Initial box size set to: {initial_box_size}")
            else:
                print("No bounding boxes detected in this frame.")

            if len(box_size_history) == box_size_history.maxlen and initial_box_size is not None:
                moving_average_box_size = np.mean(box_size_history)
                print(f"Moving average box size: {moving_average_box_size}")

                ratio = moving_average_box_size / initial_box_size
                print(f"Size ratio: {ratio}")

                if ratio < 0.75:
                    if current_model != model_visdrone:
                        current_model = model_visdrone
                        print("Switching to VisDrone model due to reduction in box size.")
                elif ratio >= 1.0:
                    if current_model != model_yolo:
                        current_model = model_yolo
                        print("Switching to YOLO model due to increase in box size.")

            timestamp = datetime.datetime.now()
            ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(night_vision_frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            processed_queue.put((night_vision_frame, None))
            frame_id += 1
        except Empty:
            continue

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
    video_path = "input/DJI_0586.MP4"

    if not use_streaming:
        print(f"Apertura del video locale: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {video_path}")
            return

    model_yolo = YOLO("yolo_weights/yolov8n.pt")
    model_visdrone = YOLO("yolo_weights/VisDrone_best.pt")

    amplification_factor = 10.0
    noise_level = 0.05
    brightness_threshold = 80
    skip_frames = 1

    frame_queue = Queue(maxsize=100)
    processed_queue = Queue(maxsize=100)

    device = get_device()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, skip_frames, (640, 480)))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold, model_yolo, model_visdrone, device))

    read_thread.start()
    process_thread.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('REAL_TIME_VIDEO.mp4', fourcc, fps, (640, 480))

    try:
        while True:
            frame_data = processed_queue.get(timeout=1)
            if frame_data[0] is None:  # Segnale di terminazione
                break
            frame, _ = frame_data
            if frame is not None and frame.size > 0:
                out.write(frame)
                cv2.imshow('Night Vision', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
    finally:
        # Chiusura sicura di tutte le risorse
        read_thread.join()
        process_thread.join()

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
