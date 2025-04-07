from flask import Flask, request, render_template, url_for
import os
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

app = Flask(__name__)

# Set the upload folder path for images inside the 'static' folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=45, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load('vit_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

@app.route('/', methods=['GET', 'POST'])
def classify_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream).convert('RGB')

            # Preprocess the image
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs).logits

            # Get predicted class index and name
            predicted_class_idx = outputs.argmax(-1).item()
            labels = [
                'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral',
                'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert',
                'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
                'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
                'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station',
                'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg',
                'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace',
                'thermal_power_station', 'wetland'
            ]
            predicted_class = labels[predicted_class_idx]

            # Save the uploaded image to the static/uploads folder
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            image.save(image_path)

            # Pass the image path relative to the 'static' folder for correct serving in HTML
            image_url = url_for('static', filename=f'uploads/{file.filename}')

            # Render the template with the result, class index, class name, and image URL
            return render_template('index.html', predicted_class=predicted_class, predicted_class_idx=predicted_class_idx, image_url=image_url)

    return render_template('index.html', predicted_class=None, predicted_class_idx=None, image_url=None)

if __name__ == "__main__":
    app.run(debug=True)
