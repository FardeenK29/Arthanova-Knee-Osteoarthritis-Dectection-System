import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = "model/koa_model.h5"
model = load_model(MODEL_PATH)

# KOA Classification Labels
CLASS_LABELS = ["Healthy", "Doubtful", "Mild", "Moderate", "Severe"]

# Function to preprocess the uploaded X-ray image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale (1 channel)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)  # Resize to match model input
    image = img_to_array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 1)
    image = np.expand_dims(image, axis=-1)  # Ensure correct shape (1, 224, 224, 1)
    return image

# Function to get KOA prediction
def predict(image):
    """
    Take an image input and return:
    - Predicted KOA condition (e.g., Healthy, Mild, Severe)
    - Model confidence score
    """
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_label = CLASS_LABELS[np.argmax(predictions)]
    confidence_score = np.max(predictions) * 100  # Convert to percentage
    return predicted_label, confidence_score

def get_model_metrics():
    """ Returns model accuracy, precision, recall, and F1-score. """
    # These values should be precomputed or retrieved dynamically if needed.
    return {
        "accuracy": 0.83,  # Example value, update with actual performance
        "precision": 0.84,
        "recall": 0.83,
        "f1_score": 0.83
    }