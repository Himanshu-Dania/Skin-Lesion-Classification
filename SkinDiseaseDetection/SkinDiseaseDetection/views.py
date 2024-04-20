# Django views.py
from django.shortcuts import render
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


get_custom_objects().update({"f1_score": f1_score})

labels = {
    0: "actinic keratosis",
    1: "basal cell carcinoma",
    2: "dermatofibroma",
    3: "melanoma",
    4: "nevus",
    5: "pigmented benign keratosis",
    6: "squamous cell carcinoma",
    7: "vascular lesion",
}

model = load_model("best_model.h5")


def predict_image(request):
    if request.method == "POST":
        image = request.FILES["image"]
        img = Image.open(image).convert("RGB")
        img = img.resize((100, 75))
        img = img.rotate(180)
        img_array = np.array(img) / 255
        img_array = np.expand_dims(img_array, axis=0)

        img_array = np.swapaxes(img_array, 1, 2)

        prediction = model.predict(img_array)
        label_index = np.argmax(prediction)
        prediction = labels[label_index]
        return render(
            request,
            "combined.html",
            {"prediction": prediction},
        )

    return render(request, "combined.html")
