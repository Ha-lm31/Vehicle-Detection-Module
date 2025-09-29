# Vehicle Detection Module \ Module de détection de véhicule
Dans ce module, je veux veux faire la détection des véhicules dans un images.
1. Sélectioner les véhicules qui viens vers la caméra,
2. classer selon leur type :
    - car (voiture),
    - truck (camion),
    - bus,
    - motorbike (moto),
    - bicycke (vélo).
3. Counter le nombre dans chaque type.
Pour faire ça, on veux utiliser des défirent modèle YOLO (You only look once).
## Modèles YOLO
Voici une listes des modéles YOLO qu'on peux utiliser : 
- yolov8s.pt
- yolov8m.pt
- yolov10s.pt
- yolov8n.pt
- yolov10n.pt
- yolov10m.pt
- yolov8l.pt
- yolov10l.pt
- yolov8x.pt
- yolov10x.pt
    - les mieux (selon moi) : 8x, 8l
    - Classement : 8x, 8l, 10m
## Implémentations

### Code1.py
Dans ce fichier, je veux un script qui appliquer un modèle YOLO `x` sur un images `y`, pour détecter et classer et compter le nombres de véhicules avec leur type qui vient vers la caméra,
retourne un images en sortier et un dictionnaire contient les informations suivantes
{"id-img": id_image, "car": 0, "truck": 0, "bus": 0, "motorbike": 0, "bicycle": 0 }

### Code2.py
Pour ce code, il fait le meme travaille du `Code1.py` mais pour tout les images du dossier.

### model.py
Le code 

