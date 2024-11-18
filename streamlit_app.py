
import socket
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

app = Flask(__name__)
CORS(app)

# Load pre-trained feature extractor (InceptionV3)
feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Load your trained captioning model
caption_model = load_model('model/model15Nov.h5')

# Load tokenizer used during training
with open('model/tokenizer15nov.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Maximum sequence length used during training
max_sequence_length = 36

def extract_features(img_path):
    """Extract features from an image using the InceptionV3 model."""
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features

def generate_caption(image_features):
    """Generate a caption for the given image features."""
    # Check if '<start>' token is in the tokenizer
    if '<start>' not in tokenizer.word_index:
        raise ValueError("The '<start>' token is missing from the tokenizer.")
    
    start_token = tokenizer.word_index['<start>']  # Start token
    sequence = [start_token]  # Initialize the sequence with the start token
    
    for _ in range(max_sequence_length):
        # Pad the sequence to the maximum length
        padded_sequence = np.zeros((1, max_sequence_length))
        padded_sequence[0, :len(sequence)] = sequence
        
        # Predict the next word
        predictions = caption_model.predict([padded_sequence, image_features])
        next_word_id = np.argmax(predictions[0])
        
        # Check if the '<end>' token is predicted
        if next_word_id == tokenizer.word_index.get('<end>', None):  # Using .get() safely
            break
        
        # Add the predicted word to the sequence
        sequence.append(next_word_id)
    
    # Convert word IDs back to words
    words = [tokenizer.index_word[id] for id in sequence if id in tokenizer.index_word]
    return ' '.join(words[1:])  # Exclude the start token

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = "./temp_image.jpg"
    file.save(file_path)

    try:
        # Extract features from the image
        image_features = extract_features(file_path)

        # Generate a caption
        caption = generate_caption(image_features)

        # Clean up the temporary file
        os.remove(file_path)

        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 5000  # Default Flask port
    print(f"Flask app running at http://{local_ip}:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
