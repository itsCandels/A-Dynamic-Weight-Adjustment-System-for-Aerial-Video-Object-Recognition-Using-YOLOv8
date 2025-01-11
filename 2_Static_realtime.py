import cv2
import numpy as np
import datetime
import time
import multiprocessing as mp
from ultralytics import YOLO
import gc
from numba import jit

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

@jit(nopython=True)
def calculate_brightness(image):
    return np.mean(image)

@jit(nopython=True)
def simulate_image_intensifier(image, amplification_factor, noise_level=0.05):
    image_float = image.astype(np.float32)
    amplified_image = image_float * amplification_factor
    noise = np.random.normal(scale=noise_level * 255, size=image.shape)
    noisy_image = amplified_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    kernel = np.ones((3, 3), np.float32) / 9
    phosphor_image = cv2.filter2D(noisy_image, -1, kernel)
    final_image = phosphor_image.astype(np.uint8)
    return final_image

def apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = calculate_brightness(gray_frame)
    night_vision_active = brightness < brightness_threshold
    if night_vision_active:
        intensified_frame = simulate_image_intensifier(gray_frame, amplification_factor, noise_level)
        intensified_frame = cv2.cvtColor(intensified_frame, cv2.COLOR_GRAY2BGR)
    else:
        intensified_frame = frame
    return intensified_frame, night_vision_active

def read_frames(video_source, frame_queue, skip_frames=1, target_size=(1920, 1080)):
    cap = cv2.VideoCapture(video_source)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame = cv2.resize(frame, target_size)
        if frame_id % skip_frames == 0:
            frame_queue.put(frame)
        frame_id += 1
    cap.release()

def process_frames(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold, model):
    while True:
        frame = frame_queue.get()
        if frame is None:
            processed_queue.put(None)
            break
        night_vision_frame, night_vision_active = apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold)
        if not night_vision_active:
            results = model(night_vision_frame)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cv2.rectangle(night_vision_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(night_vision_frame, cls_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(night_vision_frame, ts, (10, night_vision_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        processed_queue.put(night_vision_frame)

def main():
    rtmp_url = "rtmp://localhost:1935/live"
    video_path = "/Users/federicocandela/Desktop/CIFAR/DJI.MP4"
    start_time = time.time()
    timeout = 10  # Timeout di 10 secondi

    cap = cv2.VideoCapture(rtmp_url)
    while not cap.isOpened():
        if time.time() - start_time > timeout:
            print("Errore: Timeout durante la connessione allo streaming video.")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Errore: Impossibile aprire il video {video_path}")
                return
            break
        time.sleep(1)
        cap.open(rtmp_url)
    cap.release()
    
    #NORMAL
    model = YOLO("yolov8n.pt")
    #VISDRONE
    #model = YOLO("yolo_weights/VisDrone_best.pt")
    
    amplification_factor = 10.0
    noise_level = 0.1
    brightness_threshold = 20
    skip_frames = 1  # Riduci il numero di frame saltati per ridurre la latenza

    frame_queue = mp.Queue(maxsize=50)  # Limita la dimensione della coda dei frame
    processed_queue = mp.Queue(maxsize=50)  # Limita la dimensione della coda dei frame elaborati

    read_process = mp.Process(target=read_frames, args=(rtmp_url if cap.isOpened() else video_path, frame_queue, skip_frames, (640, 480)))
    process_process = mp.Process(target=process_frames, args=(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold, model))

    read_process.start()
    process_process.start()

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('/Users/federicocandela/Desktop/CIFAR/REAL_TIME_VIDEO2.mp4', fourcc, 30, (640, 480))

    while True:
        frame = processed_queue.get()
        if frame is None:
            break
        out.write(frame)
        cv2.imshow('RTMP Stream with YOLO and Night Vision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    read_process.join()
    process_process.join()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
