# Dans ce code détecte les véhicules qui viens vers la caméra dans un images.

import time
t1 = time.time()
from ultralytics import YOLO
import cv2
import os
# Charger le modèle YOLOv8
# tu peux utiliser yolov8s.pt pour plus de précision
model = YOLO("yolov8n.pt")
# Image en entrée
image_path = r"C:\Users\ot\Desktop\Model\Traffic Vehicles\train\frame_0000_jpg.rf.02488de83e72f637bed5d2fdfc2cc10b.jpg"
img = cv2.imread(image_path)
# Extraire seulement le nom du fichier
id_image = os.path.basename(image_path)
# Dictionnaire initialisé
counts = {
    "id-img": id_image,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorbike": 0,
    "bicycle": 0
}
# Détection
results = model(img)
# Récupérer dimensions de l’image pour savoir qui vient vers la caméra (côté droit)
h, w, _ = img.shape
mid_x = w // 2
# Mapping YOLO COCO classes -> tes classes
vehicle_classes = {"car": 2, "motorbike": 3, "bus": 5, "truck": 7, "bicycle": 1}
# Parcourir résultats
for r in results[0].boxes:
    cls_id = int(r.cls)
    cls_name = model.names[cls_id]
    if cls_name in vehicle_classes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        # centre de la bbox
        cx = (x1 + x2) // 2
         # garder uniquement ceux qui viennent vers caméra
        if cx > mid_x:
            counts[cls_name] += 1
            # Dessiner la box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# Sauvegarde de l'image annotée
output_path = "output_detected.jpg"
cv2.imwrite(output_path, img)
# Afficher le dictionnaire
print(counts)
t2 = time.time()
print(t2-t1)