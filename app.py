import json
import torch
import streamlit as st
from PIL import Image
from torch import nn
from torchvision import models,transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.json") as f:
    class_names=json.load(f)

num_classes=len(class_names)

model=models.shufflenet_v2_x1_0(weights=None)
model.fc=nn.Linear(model.fc.in_features,num_classes)
model.load_state_dict(torch.load("model_shufflenet_x1_0.pth",map_location=device))
model.to(device)
model.eval()

image_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict(image):
    tensor=image_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities=torch.softmax(model(tensor),dim=1)[0]

    top_probabilities,top_indices=probabilities.topk(5)
    return top_probabilities,top_indices

def show_result(image,filename=None):
    with st.spinner("Menganalisis citra..."):
        top_probabilities,top_indices=predict(image)

    predicted_class=class_names[top_indices[0].item()]
    confidence=top_probabilities[0].item()

    st.markdown("---")
    col1,col2=st.columns(2)

    with col1:
        caption=filename if filename else "Citra yang diunggah"
        st.image(image,caption=caption,use_container_width=True)

    with col2:
        st.metric("Confidence",f"{confidence:.2%}")

        if confidence>=0.8:
            st.success(f"Prediksi: {predicted_class}")
        elif confidence>=0.5:
            st.warning(f"Prediksi: {predicted_class}")
        else:
            st.error(f"Prediksi: {predicted_class} (keyakinan rendah)")

    st.write("**Top-5 Probabilitas**")

    for probability,index in zip(top_probabilities,top_indices):
        label=class_names[index.item()]
        st.write(f"{label} — {probability.item():.2%}")
        st.progress(probability.item())

    report_lines=[f"File: {filename or '-'}","Top-5 Prediksi:"]
    for probability,index in zip(top_probabilities,top_indices):
        report_lines.append(f"{class_names[index.item()]}: {probability.item():.2%}")

    st.download_button(
        "Unduh Hasil",
        "\n".join(report_lines),
        file_name=f"hasil_{filename or 'prediksi'}.txt",
        key=filename or "single"
    )

st.set_page_config(page_title="Klasifikasi Motif Batik Nitik",layout="centered")

with st.sidebar:
    st.header("Tentang Model")
    st.write("**Arsitektur:** ShuffleNetV2 x1.0")
    st.write("**Dataset:** Batik Nitik 960")
    st.write(f"**Jumlah Kelas:** {num_classes}")
    st.write("**Parameter:** 1,32 juta")
    st.write("**Ukuran Model:** 5,4 MB")

st.title("Klasifikasi Motif Batik Nitik")
uploaded_files=st.file_uploader(
    "Pilih citra batik (bisa lebih dari satu)",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Silakan unggah citra batik untuk memulai klasifikasi.")
else:
    for uploaded_file in uploaded_files:
        image=Image.open(uploaded_file).convert("RGB")
        show_result(image,filename=uploaded_file.name)

st.markdown("---")
st.caption("Sistem Klasifikasi Motif Batik Nitik | Skripsi — Fahreza Ananda Kusuma")