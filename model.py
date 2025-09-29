# === Détection véhicules avec YOLOv8 ===
import time
import os
import json
import cv2
from ultralytics import YOLO
t1 = time.time()
# === CONFIGURATION ===
# ⚠️ change ici le modèle voulu
model_path = 'yolov10x.pt'
dataset_path = 'Traffic Vehicles'
# Extraire un nom court du modèle (ex: yolov8n.pt -> v8n)
model_name = os.path.splitext(os.path.basename(model_path))[0].replace("yolo", "")
output_json = f"{model_name}.json"
output_dir = f"output_{model_name}"
os.makedirs(output_dir, exist_ok=True)
# Charger YOLO
model = YOLO(model_path)
# Classes véhicules (COCO)
vehicle_classes = {"car": 2, "motorbike": 3, "bus": 5, "truck": 7, "bicycle": 1}
# Résultats globaux
results_json = []
# Parcours train et valid
for subset in ['train', 'valid']:
    folder_path = os.path.join(dataset_path, subset)
    print(f"🔎 Lecture des images dans : {folder_path}")
    if not os.path.exists(folder_path):
        print(f"⚠️ Dossier introuvable : {folder_path}")
        continue
    for img_name in os.listdir(folder_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # ignorer les fichiers non-images
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Impossible de lire {img_path}")
            continue
        # Init dictionnaire
        counts = {
            "id-img": img_name,
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorbike": 0,
            "bicycle": 0
        }
        # Détection
        results = model(img)
        h, w, _ = img.shape
        mid_x = w // 2
        for r in results[0].boxes:
            cls_id = int(r.cls)
            cls_name = model.names[cls_id]
            if cls_name in vehicle_classes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cx = (x1 + x2) // 2
                if cx > mid_x:  # véhicules venant vers caméra
                    counts[cls_name] += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Ajouter résultats
        results_json.append(counts)
        # Sauvegarde image annotée
        cv2.imwrite(os.path.join(output_dir, f"{subset}_{img_name}"), img)
# Sauvegarde JSON final
with open(output_json, "w") as f:
    json.dump(results_json, f, indent=4)
print(f"✅ Terminé : {len(results_json)} images traitées")
print(f"✅ Résultats sauvegardés dans {output_json}")
print(f"✅ Images annotées dans {output_dir}")
t2 = time.time()
print(f"⏱ Temps total : {t2-t1:.2f} sec")
