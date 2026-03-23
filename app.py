import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = tf.keras.models.load_model('model/best_emotion_model-4.keras')
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def prepare_and_crop_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    if len(faces) > 0:
        # Largest face select karo
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

        # Bounding box draw karo
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 157), 3)
        cv2.imwrite(img_path, img)

        # RGB crop — 224x224x3 (naya model)
        roi = img[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        # Fallback — poori image use karo
        roi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ 224x224 RGB — naye model ke liye
    roi = cv2.resize(roi, (224, 224))
    roi = roi.astype('float32') / 255.0
    roi = np.reshape(roi, (1, 224, 224, 3))
    return roi


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            input_data = prepare_and_crop_image(path)
            if input_data is None:
                return render_template('index.html', error="Image load nahi hui")

            prediction = model.predict(input_data)
            label      = CLASS_NAMES[np.argmax(prediction)]
            confidence = f"{np.max(prediction)*100:.2f}%"

            return render_template('index.html',
                                   label=label.capitalize(),
                                   confidence=confidence,
                                   image=filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5500)





