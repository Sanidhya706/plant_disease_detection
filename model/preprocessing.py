from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img):
    # Resize image to the input size required by the model
    img = img.resize((224, 224))  # Adjust this size based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    
    return img_array
