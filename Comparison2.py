import time
import json
import pandas as pd

t1 = time.time()

# === Fichiers JSON générés pour chaque modèle ===
json_files = {
    "YOLOv8n": "v8n.json",
    "YOLOv8s": "v8s.json",
    "YOLOv8m": "v8m.json",
    "YOLOv8l": "v8l.json",
    "YOLOv8x": "v8x.json",
    "YOLOv10n": "v10n.json",
    "YOLOv10s": "v10s.json",
    "YOLOv10m": "v10m.json",
    "YOLOv10l": "v10l.json",
    "YOLOv10x": "v10x.json"
}

# === Charger les résultats ===
all_totals = {}
all_dataframes = {}

for model_name, file_path in json_files.items():
    try:
        with open(file_path, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Fichier introuvable : {file_path}, ignoré")
        continue
    
    df = pd.DataFrame(detections).fillna(0)
    
    # Adapter si la clé d'image diffère
    if "id-img" in df.columns:
        df.rename(columns={"id-img": "id_image"}, inplace=True)
    elif "id_image" not in df.columns:
        df["id_image"] = df.index  # fallback si pas de colonne
    
    df.set_index("id_image", inplace=True)
    all_dataframes[model_name] = df
    
    # Totaux par classe
    totals = df.sum()
    all_totals[model_name] = totals

# === Construire un tableau comparatif ===
totals_df = pd.DataFrame(all_totals).fillna(0)

print("\n=== Résumé global des détections ===")
print(totals_df)

# === Définir un critère de "meilleur modèle" ===
# Ici : somme totale des détections sur car + bus + truck
classes_of_interest = ["car", "bus", "truck"]
scores = totals_df.loc[classes_of_interest].sum()

# Trier les modèles du meilleur au moins bon
sorted_scores = scores.sort_values(ascending=False)

print("\n=== Score global (car + bus + truck) ===")
print(sorted_scores)

best_model = sorted_scores.idxmax()
print(f"\n✅ Le meilleur modèle pour détecter les véhicules est : {best_model}")

t2 = time.time()
print(f"\n⏱ Temps total : {t2-t1:.2f} sec")
