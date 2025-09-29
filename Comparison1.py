import time
import json
import glob
t1 = time.time()
# === Charger tous les fichiers JSON ===
files = glob.glob("*.json")  # prend tous les .json dans le dossier
print("🔎 Fichiers trouvés :", files)

# Charger les résultats
all_results = {}
for file in files:
    with open(file, "r") as f:
        all_results[file] = json.load(f)

# Vérifier nombre d’images (prendre le min si différent)
min_len = min(len(r) for r in all_results.values())

# === Totaux globaux ===
totals = {file: {"car":0, "truck":0, "bus":0, "motorbike":0, "bicycle":0} for file in files}
sum_totals = {file:0 for file in files}

same_results = 0
diff_results = 0

# Comparaison image par image
for i in range(min_len):
    img_counts = {}
    for file, results in all_results.items():
        img = results[i]
        img_counts[file] = {cls: img[cls] for cls in totals[file].keys()}
        for cls in totals[file].keys():
            totals[file][cls] += img[cls]

    # Vérifier si tous les modèles sont identiques pour cette image
    values = list(img_counts.values())
    if all(v == values[0] for v in values):
        same_results += 1
    else:
        diff_results += 1

# Calculer les sommes globales
for file in files:
    sum_totals[file] = sum(totals[file].values())

# === Classement des modèles du meilleur au moins bon ===
sorted_models = sorted(sum_totals.items(), key=lambda x: x[1], reverse=True)

# === Résumé ===
print("\n=== Totaux par modèle (classés) ===")
for file, total in sorted_models:
    print(f"{file}: {totals[file]} → Somme totale: {total}")

print("\n=== Comparaison ===")
print(f"Images avec mêmes résultats : {same_results}")
print(f"Images avec résultats différents : {diff_results}")

print(f"\n✅ Meilleur modèle : {sorted_models[0][0]}")
t2 = time.time()
print(f"⏱ Temps total : {t2-t1:.2f} sec")