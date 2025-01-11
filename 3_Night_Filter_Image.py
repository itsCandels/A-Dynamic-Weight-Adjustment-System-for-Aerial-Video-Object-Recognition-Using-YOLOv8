#---------------------------------------#SCURISCI IMMAGINE-----------------------------------------------
'''
# Apriamo la nuova immagine caricata
from PIL import Image, ImageEnhance

# Apriamo l'immagine caricata
image_path = "/mnt/data/1010.jpg"
image = Image.open(image_path)

# Applichiamo un filtro per scurire l'immagine e simulare la notte
enhancer = ImageEnhance.Brightness(image)
dark_image = enhancer.enhance(0.2)  # Riduciamo la luminosità

# Salviamo l'immagine modificata
dark_image_path = "/mnt/data/1010_dark.jpg"
dark_image.save(dark_image_path)

dark_image.show()
dark_image_path

#APPLICA_VISIONE_NOTTURNA
import cv2
import numpy as np

def simulate_image_intensifier(image, amplification_factor, noise_level=0.05):
    image_float = image.astype(np.float32)
    amplified_image = image_float * amplification_factor
    noise = np.random.normal(scale=noise_level * 255, size=image.shape)
    noisy_image = amplified_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    kernel = np.ones((5, 5), np.float32) / 25
    phosphor_image = cv2.filter2D(noisy_image, -1, kernel)
    final_image = phosphor_image.astype(np.uint8)
    return final_image

def calculate_brightness(image):
    return np.mean(image)

def apply_night_vision_conditionally(image, amplification_factor, noise_level, brightness_threshold):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = calculate_brightness(gray_frame)
    if brightness < brightness_threshold:
        intensified_frame = simulate_image_intensifier(gray_frame, amplification_factor, noise_level)
        intensified_frame = cv2.cvtColor(intensified_frame, cv2.COLOR_GRAY2BGR)
    else:
        intensified_frame = image
    return intensified_frame

# Parametri
amplification_factor = 10.0
noise_level = 0.1
brightness_threshold = 100  # valore aumentato per assicurare l'attivazione della visione notturna

# Carica l'immagine
image_path = "/Users/federicocandela/Desktop/YOLO_OBJECT_DETECTION/PROVa/24_dark_2.jpg"
image = cv2.imread(image_path)

# Applica la visione notturna condizionale
night_vision_image = apply_night_vision_conditionally(image, amplification_factor, noise_level, brightness_threshold)

# Salva l'immagine risultante
output_image_path = "/Users/federicocandela/Desktop/YOLO_OBJECT_DETECTION/PROVa/24_night_vision_corrected.jpg"
cv2.imwrite(output_image_path, night_vision_image)

output_image_path
'''

#---------------------------------------#FALLO_PER_VIDEO-----------------------------------------------

import cv2
import numpy as np
from PIL import Image, ImageEnhance

def darken_frame(frame, factor=0.1):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    darkened_pil_img = enhancer.enhance(factor)
    darkened_frame = cv2.cvtColor(np.array(darkened_pil_img), cv2.COLOR_RGB2BGR)
    return darkened_frame

# Parametri per il video
video_path = "/Users/federicocandela/Desktop/TEST_ALGORITMO/DJI_0590.MP4"
output_video_path = "/Users/federicocandela/Desktop/TEST_ALGORITMO/7)_VIDEO_NOTTURNA_1_PASSEGGIATA/DJI_DARK3.mp4"

# Apri il video di input
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Crea il video writer per l'output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Scurisci il frame
    darkened_frame = darken_frame(frame, factor=0.1)
    
    # Scrivi il frame nel video di output
    out.write(darkened_frame)
    
    # Visualizza il frame in tempo reale
    cv2.imshow('Darkened Frame', darkened_frame)
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed frame: {frame_count}")
    
    # Ferma l'elaborazione premendo 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video salvato in: {output_video_path}")
