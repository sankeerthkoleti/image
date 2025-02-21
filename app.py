import torch
import numpy as np
import faiss
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import base64
import io
import pickle
app = Flask(__name__)
def save_index():
    """Save FAISS index and user IDs to disk"""
    faiss.write_index(index, "faiss.index")
    with open("user_ids.pkl", "wb") as f:
        pickle.dump(user_ids, f)
# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# FAISS index for fast similarity search
dimension = 512  # CLIP feature size
index = faiss.IndexFlatL2(dimension)
image_embeddings = []
user_ids = []

def get_image_embedding(image):
    """Extract CLIP features from an image"""
    image = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**image).cpu().numpy()
    return embedding

@app.route("/add_image", methods=["POST"])
def add_image():
    data = request.json
    user_id = data["user_id"]
    image_data = base64.b64decode(data["image"])

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    embedding = get_image_embedding(image)

    global image_embeddings, index, user_ids
    image_embeddings.append(embedding)
    index.add(embedding)
    user_ids.append(user_id)

    save_index()  # 🔹 Save the index and user IDs

    return jsonify({"message": "Image added successfully"}), 200

@app.route("/find_similar", methods=["POST"])
def find_similar():
    """Find the most similar image"""
    data = request.json
    image_data = base64.b64decode(data["image"])

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    query_embedding = get_image_embedding(image)

    # ✅ Ensure FAISS index is loaded correctly
    if index is None or index.ntotal == 0:
        return jsonify({"error": "No images in database. Please upload images first."}), 400

    _, closest_index = index.search(query_embedding, 1)

    # ✅ Ensure a valid result is found
    if len(closest_index) == 0 or len(closest_index[0]) == 0:
        return jsonify({"error": "No match found"}), 400

    matched_user_id = user_ids[closest_index[0][0]]

    return jsonify({"id": matched_user_id}), 200


if __name__ == "__main__":
    app.run(port=5000)
