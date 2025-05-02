from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Equalization
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# Load your model
model = load_model('EyeDiseaseDetection.h5')  # or 'your_model.keras'

# Prepare the equalization layer
equalizer = Equalization(value_range=(0, 1))

app = FastAPI()

def preprocess_image(image: Image.Image):
    # Resize and normalize
    image = image.resize((120, 120))
    img_array = np.array(image) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Convert to tensor and apply equalization
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = equalizer(img_tensor)
    return img_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess image
    img_tensor = preprocess_image(image)

    # Predict
    prediction = model.predict(img_tensor)
    classNames = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    predicted_class = classNames[int(np.argmax(prediction, axis=1)[0])]
    return {"predicted_class": predicted_class}
