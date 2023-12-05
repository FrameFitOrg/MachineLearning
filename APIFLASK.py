# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# # Load model
# model = tf.keras.models.load_model('faceshape.h5')

# # Initialize Flask app
# app = Flask(__name__)  # Ganti _name_ menjadi __name__

# # Define API endpoint for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Menerima gambar dari permintaan POST
#         file = request.files['image']
        
#         if file is None:
#             return jsonify({'error': 'No image provided'}), 400
        
#         img = Image.open(file)  # Menggunakan Image.open() untuk membuka gambar

#         # Mengubah gambar menjadi array numpy
#         img = img.resize((150, 150))  # Mengubah ukuran gambar menjadi (150, 150)
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0  # Normalisasi

#         # Memprediksi kelas gambar
#         predictions = model.predict(img_array)
#         predictionsmax = np.argmax(predictions)
#         class_names = ['Bulat', 'Lonjong', 'Oval']  # Ganti dengan kelas yang sesuai
#         predicted_class = class_names[predictionsmax]

#         # Mengembalikan hasil prediksi dalam format JSON
#         result = {'prediction': predicted_class}
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the Flask app
# if __name__ == '__main__':  # Ganti _name_ menjadi __name__
#     app.run(host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify, send_file
# import tensorflow as tf
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
# import os

# # Load model
# model = tf.keras.models.load_model('faceshape.h5')

# # Initialize Flask app
# app = Flask(__name__)

# # Define API endpoint for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Receive image from POST request
#         file = request.files['image']

#         if file is None:
#             return jsonify({'error': 'No image provided'}), 400

#         img = Image.open(file)

#         # Convert image to numpy array
#         img = img.resize((300, 300))
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0

#         # Predict the image class
#         predictions = model.predict(img_array)
#         predictionsmax = np.argmax(predictions)
#         class_names = ['Bulat', 'Lonjong', 'Oval']
#         predicted_class = class_names[predictionsmax]

#         # Draw the predicted text on the image
#         draw = ImageDraw.Draw(img)
#         font = ImageFont.truetype("arial.ttf", 30) # Ganti font sesuai kebutuhan
#         draw.text((10, 10), f"{predicted_class}", (255, 255, 255), font=font)

#         # Save the image with prediction
#         img_path = "predicted_image.jpg"
#         img.save(img_path)  # Save the image with prediction locally

#         # Menampilkan hasil prediksi class di terminal
#         print(f"Hasil prediksi class: {predicted_class}")

#         # Return the saved image as response
#         return send_file(img_path, mimetype=f'image/{img.format.lower()}')

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

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
        img = img.resize((300, 300))  # Mengubah ukuran gambar menjadi (300, 300)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisasi

        # Memprediksi kelas gambar
        predictions = model.predict(img_array)
        predictionsmax = np.argmax(predictions)
        class_names = ['Bulat', 'Lonjong', 'Oval']  # Ganti dengan kelas yang sesuai
        predicted_class = class_names[predictionsmax]

        # Mengembalikan hasil prediksi dalam format JSON
        result = {'prediction': predicted_class}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':  # Ganti _name_ menjadi __name__
    app.run(host='0.0.0.0', port=5000)

