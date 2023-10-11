import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained model
model = tf.keras.models.load_model('faceshape_model.h5')  # Ganti 'model_path_here' dengan path menuju model Anda

# Dictionary untuk memetakan indeks ke label nama kelas
class_mapping = {
    0: 'Heart',
    1: 'Oblong',
    2: 'Oval',
    3: 'Round',
    4: 'Square',
    # Tambahkan pemetaan lainnya sesuai dengan jumlah kelas Anda
}

# Fungsi untuk mengklasifikasikan wajah
def classify_face(image):
    # Resize gambar ke ukuran yang sesuai dengan model Anda
    image = cv2.resize(image, (150, 150))  # Ubah ukuran gambar menjadi 150x150 pixels
    
    # Preprocessing gambar (misalnya, normalisasi dan perubahan format)
    image = image / 255.0  # Normalisasi gambar
    
    # Prediksi kelas
    prediction = model.predict(np.expand_dims(image, axis=0))
    
    # Ambil kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(prediction, axis=-1)
    
    # Ambil label nama kelas sesuai dengan indeks
    predicted_class_label = class_mapping.get(predicted_class_index[0], 'Unknown')
    
    return predicted_class_label

# Mulai streaming dari webcam
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    
    # Tampilkan frame dengan opsi "Capture" dan "Quit"
    cv2.putText(frame, "Press 'c' to Capture", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to Quit", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)
    
    # Tangkap gambar saat tombol 'c' ditekan
    key = cv2.waitKey(1)
    if key == ord('c'):
        # Tangkap gambar
        captured_frame = frame.copy()
        
        # Klasifikasikan wajah pada gambar yang telah ditangkap
        predicted_class = classify_face(captured_frame)
        
        # Tampilkan hasil klasifikasi pada layar
        cv2.putText(captured_frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Captured Image', captured_frame)
    
    # Keluar dari loop jika tombol 'q' ditekan
    elif key == ord('q'):
        break

# Stop streaming dan tutup jendela
cap.release()
cv2.destroyAllWindows()
