from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from retrieval import retrieve_songs_for_image
import torch

# Load your model and song database here once so they're ready when requests come in
from your_model_file import ContrastiveImageSongModel  # replace with your actual model file
import torch

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ContrastiveImageSongModel().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))

# Load song database
song_database = torch.load('song_database.pt', map_location=device)

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

    # Call retrieval function
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

if __name__ == '__main__':
    app.run(debug=True)

