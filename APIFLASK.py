from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('faceshape_model.h5')

# Initialize Flask app
app = Flask(__name__)  # Ganti _name_ menjadi __name__

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Menerima gambar dari permintaan POST
        file = request.files['image']
        
        if file is None:
            return jsonify({'error': 'No image provided'}), 400
        
        img = Image.open(file)  # Menggunakan Image.open() untuk membuka gambar

        # Mengubah gambar menjadi array numpy
        img = img.resize((224, 224))  # Mengubah ukuran gambar menjadi (224, 224)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisasi

        # Memprediksi kelas gambar
        predictions = model.predict(img_array)
        predictionsmax = np.argmax(predictions)
        class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']  # Ganti dengan kelas yang sesuai
        predicted_class = class_names[predictionsmax]

        # Mengembalikan hasil prediksi dalam format JSON
        result = {'prediction': predicted_class}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':  # Ganti _name_ menjadi __name__
    app.run(host='0.0.0.0', port=5000)
