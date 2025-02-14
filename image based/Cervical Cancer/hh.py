import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('C:/Users/Badari/OneDrive/Desktop/projects/girls/Cervical Cancer/best_model.keras')

# Define the path to your image
image_path = "C:/Users/Badari/OneDrive/Desktop/projects/girls/Cervical Cancer/data/Parabasal/cervix_pab_0006.jpg"

# Preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize to match model input size
    img_array = img_to_array(img) / 255.0  # Scale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict on a single image
def predict_image(model, image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = {0: 'Dyskeratotic', 1: 'Koilocytotic', 2: 'Metaplastic', 3: 'Normal', 4: 'Parabasal', 5: 'Superficial-Intermediate'}
    print(f'Predicted Class: {class_labels[predicted_class]}')
    print(f'Confidence Scores: {predictions[0]}')  # Shows confidence for each class

# Run prediction
predict_image(model, image_path)
