from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import project modules
from models.contrastive.model import ContrastiveImageSongModel
from inference.retrieval import retrieve_songs_for_image
import config

app = Flask(__name__, static_folder='static')
CORS(app)

# Ensure upload folder exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load checkpoint
checkpoint = torch.load(config.TRAINED_MODEL_PATH, map_location=device)

# Extract model parameters from checkpoint
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model_state = checkpoint['model_state_dict']
    embedding_dim = checkpoint.get('embedding_dim', config.DEFAULT_EMBEDDING_DIM)
    clip_model = checkpoint.get('clip_model', config.DEFAULT_CLIP_MODEL)
    song_embedding_dim = checkpoint.get('song_embedding_dim', None)
else:
    model_state = checkpoint
    embedding_dim = config.DEFAULT_EMBEDDING_DIM
    clip_model = config.DEFAULT_CLIP_MODEL
    song_embedding_dim = None  # Will determine from database

# Initialize model
model = ContrastiveImageSongModel(
    song_embedding_dim=song_embedding_dim,  # This will be updated if None
    embedding_dim=embedding_dim,
    clip_model_name=clip_model
).to(device)

model.load_state_dict(model_state)
model.eval()
print("Model loaded successfully.")

# Load song database
try:
    song_database = torch.load(config.SONG_DATABASE_PATH, map_location=device)
    print(f"Song database loaded with {len(song_database['song_ids'])} songs")
    
    # Update song_embedding_dim if it was None
    if song_embedding_dim is None and 'embeddings' in song_database:
        model.song_embedding_dim = song_database['embeddings'].size(1)
except Exception as e:
    print(f"Error loading song database: {e}")
    print("Please create a song database first with `python main.py create_db`")
    song_database = None

@app.route('/retrieve_songs', methods=['POST'])
def retrieve_songs():
    """API endpoint to retrieve songs matching an uploaded image"""
    if song_database is None:
        return jsonify({'error': 'Song database not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded image
    image_path = os.path.join(config.UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        # Call retrieval function (limiting to top 5 songs)
        results = retrieve_songs_for_image(
            model=model,
            image_path=image_path,
            song_database=song_database,
            top_k=5,
            device=device
        )

        # Prepare response
        recommended_songs = []
        for song in results:
            recommended_songs.append({
                'title': song['title'],
                'artist': song['artist'],
                'similarity': round(song['similarity'] * 100, 2)  # Convert to percentage
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
        return send_from_directory('static', 'index.html')
    
    # For accessing static files like CSS, JS, images
    try:
        return send_from_directory('static', path)
    except:
        # If the path doesn't exist, serve index.html (for client-side routing)
        return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    print(f"Server started at http://localhost:{config.PORT}/")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)