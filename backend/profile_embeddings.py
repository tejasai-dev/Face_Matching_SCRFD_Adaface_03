import os
import cv2
import json
import time
import numpy as np
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from numpy.linalg import norm

# Function to compute cosine similarity
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

# Function to extract face embedding
def get_face_embedding(image_path, detector, recognizer):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Image not found: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    faces = detector.get(img_rgb)

    if not faces:
        print(f"⚠️ No face detected in {image_path}")
        return None

    face = faces[0]  # Use first detected face
    embedding = recognizer.get(img_rgb, face)  # Extract face embedding
    return embedding

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "..", "input_images_1")  # Folder with profiles
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Profile_Output")
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "Profile_Embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)  # Ensure embeddings folder exists

# Load models
detector = FaceAnalysis(name="antelopev2", root=BASE_DIR, providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1)
recognizer = get_model(os.path.join(BASE_DIR, "models", "antelopev2", "glintr100.onnx"))
recognizer.prepare(ctx_id=-1)

# Start Timer
start_time = time.time()

# Step 1: Extract embeddings from all images
embeddings = []
image_paths = []

for profile_folder in os.listdir(INPUT_FOLDER):
    profile_path = os.path.join(INPUT_FOLDER, profile_folder)
    
    if not os.path.isdir(profile_path):
        continue  # Skip non-folder files

    for file in os.listdir(profile_path):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue  # Skip non-image files

        image_path = os.path.join(profile_path, file)
        embedding = get_face_embedding(image_path, detector, recognizer)

        if embedding is not None:
            embeddings.append(embedding)
            image_paths.append(image_path)

embeddings = np.array(embeddings)  # Convert to numpy array

# Step 2: Group Faces Using DBSCAN
if len(embeddings) >= 2:
    eps_value = 0.65  # Adjust clustering sensitivity
    clustering = DBSCAN(eps=eps_value, min_samples=1, metric="cosine").fit(embeddings)
    labels = clustering.labels_  # Each unique label represents a different person
else:
    labels = np.array([0])  # If only one image, treat it as a single person

# Step 3: Store embeddings per cluster
cluster_embeddings = {}

for idx, label in enumerate(labels):
    person_key = f"person_{label + 1}"  # Naming convention: person_1, person_2, etc.
    image_key = os.path.splitext(os.path.basename(image_paths[idx]))[0]  # Extract filename without extension

    if person_key not in cluster_embeddings:
        cluster_embeddings[person_key] = {}

    cluster_embeddings[person_key][image_key] = embeddings[idx]

# Step 4: Save each cluster's embeddings as person_X.npy
for person, embedding_dict in cluster_embeddings.items():
    save_path = os.path.join(EMBEDDINGS_DIR, f"{person}.npy")  # Save as person_1.npy, person_2.npy, etc.
    np.save(save_path, embedding_dict)
    
    formatted_keys = list(embedding_dict.keys())  # List of image keys
    print(f"✅ Saved {person}.npy with {len(embedding_dict)} embeddings (keys: {formatted_keys})")

# End Timer
end_time = time.time()
processing_time = round(end_time - start_time, 2)  # Time in seconds

# Save output details to JSON
output_details = {
    "processing_time": f"{processing_time} seconds",
    "total_images_processed": len(image_paths),
    "total_unique_clusters": len(set(labels))
}

output_json_path = os.path.join(OUTPUT_DIR, "profile_output.json")

with open(output_json_path, "w") as json_file:
    json.dump(output_details, json_file, indent=4)

print(f"\n📄 Output saved to {output_json_path}")
