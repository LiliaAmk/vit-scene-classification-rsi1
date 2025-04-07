# vit-scene-classification-rsi
# transformer-scene-classification

## Transformer-Based Scene Classification in Remote Sensing

This project applies Vision Transformers (ViT) for high-resolution remote sensing scene classification using the NWPU-RESISC45 dataset. It includes model training, optimization, and deployment through a web application.

## Objectives

- Train and fine-tune a pre-trained ViT model on the NWPU-RESISC45 dataset.
- Apply model optimization techniques such as pruning, quantization, or knowledge distillation.
- Deploy the optimized model using a Flask web interface.

## References

- https://github.com/VIROBO-15/Transformer-in-Remote-Sensing
- https://www.mdpi.com/2072-4292/16/13/2323
- https://link.springer.com/article/10.1007/s11063-024-11451-0
- https://ieeexplore.ieee.org/document/9553684

## Dataset

- Name: NWPU-RESISC45
- Classes: 45
- Images per class: 700
- Image size: 256×256

## Model

- Backbone: `google/vit-base-patch16-224`
- Training framework: PyTorch
- Fine-tuned for 5 epochs
- Best validation accuracy: 93.51%

## Optimization (Upcoming)

Techniques to be applied:
- Pruning
- Quantization
- Knowledge Distillation

Tools: ONNX, TensorFlow Lite, OpenVINO

## Web Application

A Flask-based web app is included to classify uploaded satellite images.

### How to Run

1. Clone the repository
2. Install dependencies : pip install -r requirements.txt
3. Launch the app:python app.py
-- Open a browser and go to `http://localhost:5000`

## File Structure

- `vit_model.pth`: Trained ViT model weights
- `app.py`: Flask application code
- `index.html`: Simple front-end interface
- `requirements.txt`: Python dependencies
