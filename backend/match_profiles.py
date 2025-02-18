import os
import numpy as np
import time
import json
import shutil
from numpy.linalg import norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define Directories
Gallery_images_dir = os.path.join(BASE_DIR, "..", "Gallery")  # Original images
Gallery_embeddings_dir = os.path.join(BASE_DIR, "..", "Gallery_Output", "Gallery_Embeddings")
Profile_embeddings_dir = os.path.join(BASE_DIR, "..", "Profile_Output", "Profile_Embeddings")
Matching_Output_Dir = os.path.join(BASE_DIR, "..", "matching_Output")
Matched_Photos_Dir = os.path.join(Matching_Output_Dir, "Matched_Photos")

# Create output directories if they don't exist
os.makedirs(Matching_Output_Dir, exist_ok=True)
os.makedirs(Matched_Photos_Dir, exist_ok=True)

# Cosine similarity function
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))

# Function to find the correct file extension
def find_correct_extension(base_filename):
    """Check and return the correct file path from the Gallery folder."""
    for ext in [".jpg", ".jpeg", ".png"]:  # Possible extensions
        file_path = os.path.join(Gallery_images_dir, base_filename + ext)
        if os.path.exists(file_path):
            return file_path  # Return the correct file path
    return None  # No matching file found

# Start Timer
start_time = time.time()

# Step 1: Load Gallery Embeddings
gallery_embeddings = {}
for file in os.listdir(Gallery_embeddings_dir):
    if file.endswith(".npy"):
        file_path = os.path.join(Gallery_embeddings_dir, file)
        data = np.load(file_path, allow_pickle=True)

        person_name = file.replace(".npy", "")

        if isinstance(data, np.ndarray) and data.shape == ():
            gallery_embeddings[person_name] = data.item()
        elif isinstance(data, np.ndarray):
            gallery_embeddings[person_name] = {"embedding": data}
        elif isinstance(data, dict):
            gallery_embeddings[person_name] = data
        else:
            print(f"⚠️ Warning: Unrecognized format in {file}")
print("✅ Successfully loaded the Gallery Embeddings!")

# Step 2: Load Profile Embeddings
profile_embeddings = {}
for file in os.listdir(Profile_embeddings_dir):
    if file.endswith(".npy"):
        file_path = os.path.join(Profile_embeddings_dir, file)
        data = np.load(file_path, allow_pickle=True)

        profile_name = file.replace(".npy", "")

        if isinstance(data, np.ndarray) and data.shape == ():
            profile_embeddings[profile_name] = data.item()
        elif isinstance(data, np.ndarray):
            profile_embeddings[profile_name] = {'embedding': data}
        elif isinstance(data, dict):
            profile_embeddings[profile_name] = data
        else:
            print(f"⚠️ Warning: Unrecognized format in {file}")
print("✅ Successfully loaded the Profile Embeddings!")

# Step 3: Match Profiles with Gallery Images
Match_Threshold = 0.45
matched_results = {}

matching_start_time = time.time()

for profile_name, profile_embeds in profile_embeddings.items():
    matched_faces = set()  # Use a set to store unique matched images
    profile_dir = os.path.join(Matched_Photos_Dir, profile_name)
    os.makedirs(profile_dir, exist_ok=True)

    for profile_key, profile_emb in profile_embeds.items():
        for gallery_name, gallery_embeds in gallery_embeddings.items():
            gallery_emb = gallery_embeds.get("embedding", None)

            if gallery_emb is not None:
                similarity = cosine_similarity(profile_emb, gallery_emb)

                if similarity > Match_Threshold:
                    base_filename = gallery_name.split("_face")[0]
                    gallery_image_path = find_correct_extension(base_filename)

                    if gallery_image_path and gallery_image_path not in matched_faces:
                        matched_faces.add(gallery_image_path)  # Add unique paths only

                        print(f"✅ Match Found! {profile_key} ↔ {gallery_name} [Similarity: {similarity:.4f}]")

                        matched_image_path = os.path.join(profile_dir, os.path.basename(gallery_image_path))
                        shutil.copy(gallery_image_path, matched_image_path)
    matched_results[profile_name] = list(matched_faces)  # Convert set to list for JSON serialization

matching_end_time = time.time()
total_time = matching_end_time - start_time
matching_time = matching_end_time - matching_start_time

# Step 4: Save Matches to JSON
matches_json_path = os.path.join(Matching_Output_Dir, "matches.json")
with open(matches_json_path, "w") as json_file:
    json.dump(matched_results, json_file, indent=4)

print("\n📄 Final Matched Results:")
for profile, matched_groups in matched_results.items():
    print(f"{profile} matches with: {matched_groups}")

print(f"\n✅ Matches saved to {matches_json_path}")
print(f"\n⏳Total Execution Time: {total_time:.4f} seconds")
print(f"⏳Time taken for Face Matching: {matching_time:.4f} seconds")
