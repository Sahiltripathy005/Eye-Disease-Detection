import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from PIL import Image

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'] 

@st.cache_resource
def load_model():
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling='max'
    )
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006),
              activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(4, activation='softmax')  
    ])
    model.load_weights('Deploy_Streamlit\\efficientnetb3-Eye Disease-weights.h5')
    model.compile(optimizer=Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = load_model()

st.title("Eye Disease Classification")
st.write("Upload an eye image to classify it into one of the following categories:")
st.write(", ".join(class_names))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=400)

    image = image.resize((224, 224))
    image_array = np.array(image) 
    image_array = np.expand_dims(image_array, axis=0)  

    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Predicted Class: **{predicted_class}**")