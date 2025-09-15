# detect_faces_from_filename.py
import os
import shutil

from insightface.app import FaceAnalysis
import cv2
import datetime
import pandas as pd
import sys
import matplotlib.pyplot as plt

def init_face_app():
    """Initialise une seule fois le modèle InsightFace."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(1280, 1280))
    return app


def detect_faces_on_image(image_path, app, show_image=True, save_faces=False, save_face_dir="faces"):
    """
    Détecte les visages sur une image de groupe avec précision.
    
    Params:
        image_path (str): Chemin vers l'image.
        app (FaceAnalysis): Modèle InsightFace initialisé.
        show_image (bool): Si True, affiche l'image avec les visages détectés.
        save_faces (bool): Si True, enregistre chaque visage détecté.
        save_dir (str): Dossier de sauvegarde des visages.

    Returns:
        int: Nombre de visages détectés.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas.")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger {image_path} avec OpenCV.")

    # Détection des visages
    faces = app.get(img)

    # Filtrage des chevauchements (IoU > 0.5)
    filtered_faces = []
    for f in faces:
        overlap = False
        for ff in filtered_faces:
            xA = max(f.bbox[0], ff.bbox[0])
            yA = max(f.bbox[1], ff.bbox[1])
            xB = min(f.bbox[2], ff.bbox[2])
            yB = min(f.bbox[3], ff.bbox[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
            boxBArea = (ff.bbox[2]-ff.bbox[0])*(ff.bbox[3]-ff.bbox[1])
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > 0.5:
                overlap = True
                break
        if not overlap:
            filtered_faces.append(f)

    num_faces = len(filtered_faces)
    print(f"Nombre de visages détectés (après filtrage) : {num_faces}")

    # Sauvegarde des visages détectés
    if save_faces and num_faces > 0:
        os.makedirs(save_face_dir, exist_ok=True)
        for i, face in enumerate(filtered_faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(save_face_dir, f"{os.path.basename(image_path)}_face_{i+1}.jpg"), face_crop)

    # Dessiner les cadres
    if show_image:
        for face in filtered_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"{num_faces} visages détectés")

        # Dossier de sauvegarde
        save_img_dir = "imagesAnalysees"
        os.makedirs(save_img_dir, exist_ok=True)

        # Nom de fichier uniquement
        filename = os.path.basename(image_path)
        full_path = os.path.join(save_img_dir, filename)

        plt.savefig(full_path)
        plt.close() 

    return num_faces


def parse_date_from_filename(filename):
    """Extrait la date du nom de fichier (format JJ-MM-AAAA)."""
    base = os.path.splitext(filename)[0]  # enlève extension
    try:
        date_obj = datetime.datetime.strptime(base, "%d-%m-%Y")
    except ValueError:
        return None, None  # nom invalide

    # Conversion format YYYY/MM/DD
    date_str = date_obj.strftime("%Y/%m/%d")

    # Jour de la semaine en français
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    jour_semaine = jours[date_obj.weekday()]

    return date_str, jour_semaine


if __name__ == "__main__":
    # Dossiers utilisés
    input_dir = "entree"
    archive_dir = "ancienne"
    save_face_dir = "faces"
    save_img_dir = "imagesAnalysees"
    csv_path = "effectifs.csv"

    # Création des dossiers si nécessaire
    for d in [input_dir, archive_dir, save_face_dir, save_img_dir]:
        os.makedirs(d, exist_ok=True)

    # Initialisation modèle InsightFace
    app = init_face_app()

    data = []

    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, file)

            # Extraire la date du nom (JJ-MM-AAAA)
            date_str, jour = parse_date_from_filename(file)
            if not date_str:
                print(f"⚠️ Nom invalide pour la date : {file}")
                continue

            # Détection des visages
            nb_faces = detect_faces_on_image(image_path, app, show_image=True, save_faces=False, save_face_dir=save_face_dir)

            print(f"{file} → {date_str} ({jour}) | {nb_faces} visages")

            data.append([date_str, jour, nb_faces])

            # Déplacement du fichier dans "ancienne"
            shutil.move(image_path, os.path.join(archive_dir, file))

    # Sauvegarde CSV
    if data:
        header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
        df = pd.DataFrame(data, columns=["date", "jour", "nombre_personnes"])
        df.to_csv(csv_path, mode='a', header=header, index=False)

    sys.exit(0)