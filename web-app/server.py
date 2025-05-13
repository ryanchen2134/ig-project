from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
from PIL import Image
import sys

# Add the backend directory to the path so we can import from it
sys.path.append('./backend')
from retrieval import retrieve_songs_for_image
from models.decoder.contrastive import ContrastiveImageSongModel

app = Flask(__name__, static_folder='frontend')
CORS(app)

UPLOAD_FOLDER = './uploads'
MODEL_PATH = './backend/contrastive_model_resnet18_locked_44.57.pth'
SONG_DB_PATH = './backend/test_song_database.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Extract model parameters from checkpoint
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_state = checkpoint['model_state_dict']
    embedding_dim = checkpoint.get('embedding_dim', 64)
    backbone_type = checkpoint.get('backbone_type', 'resnet18')
    song_embedding_dim = checkpoint.get('song_embedding_dim', 64)
else:
    model_state = checkpoint
    embedding_dim = 64
    backbone_type = 'resnet18'
    song_embedding_dim = 6144  # Default if not specified

# Initialize model with the correct parameters
model = ContrastiveImageSongModel(
    song_embedding_dim=song_embedding_dim,
    embedding_dim=embedding_dim,
    backbone_type=backbone_type
).to(device)

model.load_state_dict(model_state)
model.eval()
print("Model loaded successfully.")

# Load song database
try:
    song_database = torch.load(SONG_DB_PATH, map_location=device)
    print(f"Song database loaded with {len(song_database['song_ids'])} songs")
except Exception as e:
    print(f"Error loading song database: {e}")
    # If database doesn't exist, you could create it here 
    # (but we'll assume it exists for simplicity)

@app.route('/retrieve_songs', methods=['POST'])
def retrieve_songs():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        # Call retrieval function (limiting to top 5 songs)
        results = retrieve_songs_for_image(model, image_path, song_database, top_k=5, device=device)

        # Prepare response
        recommended_songs = []
        for song in results:
            recommended_songs.append({
                'title': song['title'],
                'artist': song['artist'],
                'similarity': round(song['similarity'] * 100, 2)  # convert to percentage
            })

        return jsonify({'recommended_songs': recommended_songs})
    except Exception as e:
        print(f"Error in retrieval: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Serve static files from the frontend directory
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path == "":
        # Serve index.html for the root endpoint
        return send_from_directory('frontend', 'index.html')
    
    # For accessing the static files like CSS, JS, images
    try:
        return send_from_directory('frontend', path)
    except:
        # If the path doesn't exist, serve index.html (for client-side routing)
        return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    print("Server started at http://localhost:5000/")
    app.run(debug=True)