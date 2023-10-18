import requests

def send_image(url, image_path):
    try:
        # Membuka file gambar
        with open(image_path, 'rb') as file:
            # Membuat permintaan POST
            response = requests.post(url, files={'image': file})
            
            # Menampilkan respons dari server
            print(response.json())
    except Exception as e:
        print(str(e))

# URL endpoint untuk mengirim gambar
url = 'https://vlm2jkrd-5000.asse.devtunnels.ms/predict'  # Ganti dengan URL API Anda

# Path gambar yang akan dikirim
image_path = '7.jpg'  # Ganti dengan path gambar yang sesuai

# Mengirim gambar ke server
send_image(url, image_path)
