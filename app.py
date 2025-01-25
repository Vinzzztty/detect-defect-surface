import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt


# Load the model
@st.cache_resource
def load_model():
    model = efficientnet_b0(pretrained=False)

    # Modify the classifier to match the training configuration
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),  # Intermediate layer
        nn.ReLU(),
        nn.Dropout(0.5),  # Regularization
        nn.Linear(256, 4),  # Output layer for 4 classes
    )

    state_dict = torch.load(
        "models/Best Model EfficientNet B0.pth", map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5], [0.5]
            ),  # Adjust if your model uses different normalization
        ]
    )

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  # Get the predicted class index
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Get probabilities
    return predicted.item(), probabilities.squeeze().tolist()


defect_classes = ["Hole", "Line", "Stain", "Thread"]  # Updated class labels

# Streamlit interface
st.title("Deteksi Kerusakan Permukaan Kain")
st.write("Unggah gambar untuk mendeteksi kerusakan pada permukaan kain.")

# Add project information
st.write("### Tentang Proyek Ini")
st.write(
    "Proyek ini menggunakan model deep learning berbasis EfficientNet-B0 untuk mendeteksi jenis kerusakan pada permukaan kain. "
    'Model ini dilatih menggunakan dataset yang berisi 4 jenis kerusakan: "Hole", "Line", "Stain", dan "Thread". '
    "Tujuannya adalah untuk membantu proses inspeksi kualitas kain secara otomatis."
)

# Add experimental results
st.write("### Hasil Eksperimen")
st.markdown(
    """
    | No | Nama Model                    | Akurasi              | Precision           | Recall              | F1 Score            |
    |----|-------------------------------|----------------------|---------------------|---------------------|---------------------|
    | 1  | YoloV5 + CSPDarknet53        | 0.288                | 0.332               | 0.26                | -                   |
    | 2  | Simple CNN                   | Train: 73.38%<br>Test: - | Train: 0.7386<br>Test: 0.7622 | Train: 0.7338<br>Test: 0.7595 | Train: 0.7291<br>Test: 0.7564 |
    | 3  | CNN + ResNet18               | Train: 95.03%<br>Test: 75.95% | Train: 0.9504<br>Test: 0.7798 | Train: 0.9503<br>Test: 0.7798 | Train: 0.9503<br>Test: 0.7750 |
    | 4  | Fine-Tunning CNN + ResNet18  | Train: 91.97%<br>Test: 77.98% | Train: 0.9208<br>Test: 0.7798 | Train: 0.9197<br>Test: 0.7798 | Train: 0.9197<br>Test: 0.7750 |
    | 5  | CNN + ResNet50               | Train: 93.25%<br>Test: 80.23% | Train: 0.9331<br>Test: 0.7995 | Train: 0.9325<br>Test: 0.8023 | Train: 0.9325<br>Test: 0.7995 |
    | 6  | CNN + EfficientNet-B0        | Train: 95.43%<br>Test: 80.34% | Train: 0.9553<br>Test: 0.8036 | Train: 0.9543<br>Test: 0.8034 | Train: 0.9543<br>Test: 0.7992 |
    | 7  | CNN + EfficientNet-B3        | Train: 98.31%<br>Test: 79.85% | Train: 0.9832<br>Test: 0.7978 | Train: 0.9831<br>Test: 0.7985 | Train: 0.9831<br>Test: 0.7965 |
    """,
    unsafe_allow_html=True,
)

model = load_model()

uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    st.write("Mendeteksi kelas kerusakan...")
    predicted_class, probabilities = predict_image(model, image)
    st.write(f"Kelas yang Diprediksi: {defect_classes[predicted_class]}")

    st.write("Probabilitas untuk Setiap Kelas:")
    for i, prob in enumerate(probabilities):
        st.write(f"{defect_classes[i]}: {prob * 100:.2f}%")
