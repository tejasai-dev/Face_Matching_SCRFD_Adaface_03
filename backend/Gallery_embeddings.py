import os
import cv2
import numpy as np
import time
import json
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "antelopev2")
GALLERY_PATH = os.path.join(BASE_DIR, "..", "Gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "Gallery_Output")
EMBEDDINGS_PATH = os.path.join(OUTPUT_PATH, "Gallery_Embeddings")
PROCESSED_IMAGES_FILE = os.path.join(OUTPUT_PATH, "processed_images.json")
MAPPING_FILE = os.path.join(OUTPUT_PATH, "embeddings_mapping.json")

os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

# Load processed images if available
if os.path.exists(PROCESSED_IMAGES_FILE):
    with open(PROCESSED_IMAGES_FILE, "r") as f:
        processed_images = set(json.load(f))
else:
    processed_images = set()

# Model files
SCRFD_MODEL_PATH = os.path.join(MODEL_PATH, "scrfd_10g_bnkps.onnx")
ADA_MODEL_PATH = os.path.join(MODEL_PATH, "glintr100.onnx")

if not os.path.exists(SCRFD_MODEL_PATH):
    raise FileNotFoundError(f"SCRFD model not found at: {SCRFD_MODEL_PATH}")
if not os.path.exists(ADA_MODEL_PATH):
    raise FileNotFoundError(f"AdaFace model not found at: {ADA_MODEL_PATH}")

# Load models
detector = FaceAnalysis(name="antelopev2", root=BASE_DIR, providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1)
recognizer = model_zoo.get_model(ADA_MODEL_PATH)
recognizer.prepare(ctx_id=-1)

# Get the last used global face index
existing_embeddings = [f for f in os.listdir(EMBEDDINGS_PATH) if f.endswith(".npy")]
last_face_index = 0

if existing_embeddings:
    indices = [
        int(f.split("_")[-1].split(".")[0]) for f in existing_embeddings if f.split("_")[-1].split(".")[0].isdigit()
    ]
    last_face_index = max(indices) + 1 if indices else 0

# Counters and timer
total_faces = 0
group_images_count = 0
start_time = time.time()
embeddings_mapping = {}

# Process each image in the gallery folder
for image_name in os.listdir(GALLERY_PATH):
    if image_name in processed_images:
        continue  # Skip already processed images

    gallery_image_path = os.path.join(GALLERY_PATH, image_name)
    img = cv2.imread(gallery_image_path)

    if img is None:
        print(f"Skipping unreadable image: {image_name}")
        continue

    faces = detector.get(img)

    if not faces:
        print(f"No faces detected in {image_name}")
        continue

    group_images_count += 1

    for i, face in enumerate(faces):
        embedding = recognizer.get(img, face)
        image_base_name = os.path.splitext(image_name)[0]
        face_filename = f"{image_base_name}_face_{i}_{last_face_index}.npy"
        np.save(os.path.join(EMBEDDINGS_PATH, face_filename), embedding)

        embeddings_mapping[face_filename] = os.path.join(GALLERY_PATH,image_name)
        last_face_index += 1
        total_faces += 1

    processed_images.add(image_name)

with open(MAPPING_FILE, "w") as f:
    json.dump(embeddings_mapping, f, indent=4)

# Save processed image names
with open(PROCESSED_IMAGES_FILE, "w") as f:
    json.dump(list(processed_images), f, indent=4)

end_time = time.time()
processing_time = end_time - start_time

print(f"Total faces processed: {total_faces}")
print(f"Total group images processed: {group_images_count}")
print(f"Time taken for processing gallery: {processing_time:.4f} seconds")

# Save output.json with processing details
output_details = {
    "processing_time": f"{processing_time:.4f} Seconds",
    "total_faces": total_faces,
    "total_group_images": group_images_count
}

output_json_path = os.path.join(OUTPUT_PATH, "output.json")
with open(output_json_path, "w") as outfile:
    json.dump(output_details, outfile, indent=4)

print(f"Output details saved to {output_json_path}")
