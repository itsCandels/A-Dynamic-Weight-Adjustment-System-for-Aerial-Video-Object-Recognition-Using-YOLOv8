'''
import cv2
import numpy as np
import time
import threading
#_--------------------------------------------------- GREEN VERSION --------------------------------------------
# Funzione per applicare il filtro di visione notturna
def apply_night_vision(frame):
    noise = np.random.normal(0, 25, frame.shape)
    noisy_frame = frame + noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)
    colored_frame = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
    green_channel = cv2.merge([np.zeros_like(gray), gray, np.zeros_like(gray)])
    night_vision_frame = cv2.addWeighted(colored_frame, 0.5, green_channel, 0.5, 0)
    return night_vision_frame

def read_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.append(None)
            break
        frame_queue.append(frame)

def process_frames(frame_queue, processed_queue, scale_percent, width, height):
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    while True:
        if not frame_queue:
            continue
        frame = frame_queue.pop(0)
        if frame is None:
            processed_queue.append(None)
            break
        small_frame = cv2.resize(frame, (new_width, new_height))
        night_vision_frame_small = apply_night_vision(small_frame)
        night_vision_frame = cv2.resize(night_vision_frame_small, (width, height))
        processed_queue.append(night_vision_frame)

# Apri il video
cap = cv2.VideoCapture('/Users/federicocandela/Desktop/YOLO_OBJECT_DETECTION/PROVA_BOCALE.mp4')

# Ottieni informazioni sul video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Crea il video writer
out = cv2.VideoWriter('output_night_vision.mp4', fourcc, fps, (width, height))

# Code for multi-threading
frame_queue = []
processed_queue = []

# Avvia i thread per la lettura e l'elaborazione dei frame
read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue))
process_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_queue, 50, width, height))

start_time = time.time()

read_thread.start()
process_thread.start()

while True:
    if not processed_queue:
        continue
    frame = processed_queue.pop(0)
    if frame is None:
        break
    out.write(frame)
    cv2.imshow('Night Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

read_thread.join()
process_thread.join()

end_time = time.time()

processing_time = end_time - start_time
video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

print(f"Tempo di elaborazione: {processing_time:.2f} secondi")
print(f"Lunghezza del video: {video_length:.2f} secondi")

# Rilascia le risorse
cap.release()
out.release()
cv2.destroyAllWindows()

'''
import cv2
import numpy as np

def simulate_image_intensifier(image, amplification_factor, noise_level=0.05):
    """
    Simula un tubo ad intensificazione di immagine.
    
    :param image: L'immagine originale in scala di grigi.
    :param amplification_factor: Il fattore di amplificazione della luce.
    :param noise_level: Livello di rumore aggiunto.
    :return: L'immagine intensificata.
    """
    # Step 1: Fotocatodo - convertiamo l'immagine in float per evitare sovraccarichi
    image_float = image.astype(np.float32)
    
    # Step 2: Moltiplicazione elettronica - amplifichiamo l'immagine
    amplified_image = image_float * amplification_factor
    
    # Step 3: Aggiungere rumore (simula imperfezioni e variazioni nel processo di moltiplicazione)
    noise = np.random.normal(scale=noise_level * 255, size=image.shape)
    noisy_image = amplified_image + noise
    
    # Step 4: Clipping per mantenere i valori dei pixel nell'intervallo 0-255
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # Step 5: Schermo fosforescente - ridurre leggermente la nitidezza per simulare l'effetto del fosforo
    kernel = np.ones((5,5),np.float32)/25
    phosphor_image = cv2.filter2D(noisy_image, -1, kernel)
    
    # Convertiamo l'immagine di nuovo in uint8
    final_image = phosphor_image.astype(np.uint8)
    
    return final_image

# Carichiamo un'immagine a bassa luminosità
input_image = cv2.imread('/Users/federicocandela/Desktop/9)RILEVATORE_MOVIMENTO_VIDEO/database/IMG_PROVA.png', cv2.IMREAD_GRAYSCALE)

# Impostiamo il fattore di amplificazione e il livello di rumore
amplification_factor = 10.0
noise_level = 0.1

# Intensifichiamo l'immagine
intensified_image = simulate_image_intensifier(input_image, amplification_factor, noise_level)

# Mostriamo l'immagine originale e quella intensificata
cv2.imshow('Original Image', input_image)
cv2.imshow('Intensified Image', intensified_image)

# Attendi la pressione di un tasto per chiudere le finestre
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
'''

import cv2
import numpy as np
import time
import threading
from queue import Queue

def simulate_image_intensifier(image, amplification_factor, noise_level=0.05):
    image_float = image.astype(np.float32)
    amplified_image = image_float * amplification_factor
    noise = np.random.normal(scale=noise_level * 255, size=image.shape)
    noisy_image = amplified_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    kernel = np.ones((5,5),np.float32)/25
    phosphor_image = cv2.filter2D(noisy_image, -1, kernel)
    final_image = phosphor_image.astype(np.uint8)
    return final_image

def calculate_brightness(image):
    return np.mean(image)

def apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = calculate_brightness(gray_frame)
    if (brightness < brightness_threshold):
        intensified_frame = simulate_image_intensifier(gray_frame, amplification_factor, noise_level)
        intensified_frame = cv2.cvtColor(intensified_frame, cv2.COLOR_GRAY2BGR)
    else:
        intensified_frame = frame
    return intensified_frame

def read_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)

def process_frames(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold):
    while True:
        frame = frame_queue.get()
        if frame is None:
            processed_queue.put(None)
            break
        night_vision_frame = apply_night_vision_conditionally(frame, amplification_factor, noise_level, brightness_threshold)
        processed_queue.put(night_vision_frame)

# Apri il video
cap = cv2.VideoCapture('/Users/federicocandela/Downloads/PROVA_BOCALE.mp4')

# Ottieni informazioni sul video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Crea il video writer
out = cv2.VideoWriter('output_night_vision.mp4', fourcc, fps, (width, height))

# Code for multi-threading
frame_queue = Queue()
processed_queue = Queue()

# Impostazioni del fattore di amplificazione e il livello di rumore
amplification_factor = 10.0
noise_level = 0.1

# Soglia per attivare la visione notturna
brightness_threshold = 20

# Avvia i thread per la lettura e l'elaborazione dei frame
read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue))
process_thread = threading.Thread(target=process_frames, args=(frame_queue, processed_queue, amplification_factor, noise_level, brightness_threshold))

start_time = time.time()

read_thread.start()
process_thread.start()

while True:
    frame = processed_queue.get()
    if frame is None:
        break
    out.write(frame)
    cv2.imshow('Night Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

read_thread.join()
process_thread.join()

end_time = time.time()

processing_time = end_time - start_time
video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

print(f"Tempo di elaborazione: {processing_time:.2f} secondi")
print(f"Lunghezza del video: {video_length:.2f} secondi")

# Rilascia le risorse
cap.release()
out.release()
cv2.destroyAllWindows()





