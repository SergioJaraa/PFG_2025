import streamlit as st
from PIL import Image
import os
import tempfile
import torch
import pickle
import numpy as np
from evaluate import load_model, preprocess_image, classify_image
import torch.nn.functional as F
import pandas as pd
from huggingface_hub import hf_hub_download
import urllib.request
import sys

sys.path.append("D:/PFG/PFG_2025/Streamlit/stylegan2-ada-pytorch")

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = "ready"
if "results" not in st.session_state:
    st.session_state.results = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "current_notes" not in st.session_state:
    st.session_state.current_notes = ""
if "current_label" not in st.session_state:
    st.session_state.current_label = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = ""
if "classified_once" not in st.session_state:
    st.session_state.classified_once = False

# Page setup
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

# CSS styles
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0d1117;
            color: #d1d5db;
        }

        .main > div:first-child {
            padding-top: 2rem;
        }

        .title {
            margin-top: 20px;
            font-size: 48px;
            margin-bottom: 30px;
            font-weight: 300;
            text-align: center;
            color: #f1faee;
            font-family: 'Courier New', Courier, monospace;
            padding: 10px;
        }

        .section-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #f1faee;
            margin-top: 20px;
            margin-bottom: 15px;
            font-family: 'Courier New', Courier, monospace;
            padding: 5px;
            padding-bottom: 10px;
            padding-top: 10px;
            margin-left: 10px;
        }

        .block-container {
            padding: 5rem;
        }

        /* All buttons */
        button {
            background-color: #30363d;
            color: #f0f6fc;
            border: 1px solid #58a6ff;
            padding: 0.5em 1.1em;
            border-radius: 8px;
            font-size: 1.05rem;
            font-weight: 500;
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }

        button:hover {
            background-color: #58a6ff;
            color: #0d1117;
            transform: scale(1.03);
            cursor: pointer;
        }

        /* File uploader */
        section[data-testid="stFileUploader"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 1em;
            border-radius: 10px;
        }

        /* Selectbox dropdown */
        div[data-baseweb="select"] {
            background-color: #161b22 !important;
            border-radius: 8px;
            border: 1px solid #30363d;
        }

        div[data-baseweb="select"] > div {
            color: #f0f6fc;
        }

        /* Text area */
        textarea {
            background-color: #161b22 !important;
            color: #f0f6fc !important;
            border-radius: 6px !important;
            border: 1px solid #30363d !important;
        }

        /* Result prediction box */
        .result {
            background-color: #1c2128;
            border: 1px solid #58a6ff;
            padding: 14px;
            border-radius: 10px;
            text-align: center;
            color: #f1faee;
            font-size: 24px;
            margin-top: 20px;
        }

        /* Radio buttons alignment */
        .stRadio > div {
            justify-content: center;
        }

        /* Expander box */
        .st-expander {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
        }
    </style>
""", unsafe_allow_html=True)    

# Title
st.markdown("<div class='title'>🧠 Brain Tumor Research Interface</div>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Class names and info
class_labels = ["Glioma", "Meningioma", "No_Tumor", "Pituitary"]
class_info = {
    "Glioma": "Tumor from glial cells. Can be malignant.",
    "Meningioma": "Usually benign tumor in the meninges.",
    "Pituitary": "Tumor in the pituitary gland. Affects hormones.",
    "No_Tumor": "No tumor detected in the MRI."
}

# Layout
col1, _, col2 = st.columns([1, 0.1, 1])

# Classify section
with col1:
    st.markdown("<div class='section-title'>🔬 Classify Tumor Image</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size: 26px;'>Test your knowledge trying to guess the tumor and check with the Federated Learning CNN Model trained by three clients</p>", unsafe_allow_html=True)

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload brain MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key=f"file_uploader_{st.session_state.uploader_key}")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("L")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Reset saved flag on new image
            if st.session_state.current_filename != uploaded_file.name:
                st.session_state.saved = False
                st.session_state.classified_once = False

            if not st.session_state.classified_once:
                with st.form(key="classification_form"):
                    user_guess = st.radio("YOUR GUESS", class_labels, horizontal=True)
                    submit_classify = st.form_submit_button("Classify")

                    if submit_classify:
                        input_tensor = preprocess_image(image)
                        outputs = model(input_tensor)
                        probs = F.softmax(outputs, dim=1).squeeze().tolist()
                        pred_idx = int(torch.argmax(outputs))
                        label = class_labels[pred_idx]

                        st.session_state.current_label = label
                        st.session_state.current_filename = uploaded_file.name
                        st.session_state.classified_once = True
                        st.session_state.probs = probs
                        st.session_state.user_guess = user_guess

                        st.rerun()

            elif st.session_state.classified_once:
                # Mostrar resultados después de clasificar
                st.markdown(f"<div class='result'>Prediction: <b>{st.session_state.current_label}</b></div>", unsafe_allow_html=True)

                if st.session_state.user_guess == st.session_state.current_label:
                    st.success("✅ Your guess was correct!")
                else:
                    st.error(f"❌ Your guess was {st.session_state.user_guess}. Model predicted {st.session_state.current_label}.")

                st.markdown("### Prediction Probabilities:")
                for i, prob in enumerate(st.session_state.probs):
                    st.write(f"- {class_labels[i]}: **{prob:.2%}**")

                if st.session_state.current_label in class_info:
                    st.info(class_info[st.session_state.current_label])

                if not st.session_state.saved:
                    with st.expander("📝 Add clinical notes"):
                        st.session_state.current_notes = st.text_area("Notes", height=150, placeholder="Write observations or notes here...", value=st.session_state.current_notes)

                    if st.button("💾 Save Notes & Prediction", use_container_width=True):
                        notes = st.session_state.current_notes if st.session_state.current_notes else ""
                        st.session_state.results.append((st.session_state.current_filename, st.session_state.current_label, notes))
                        st.success("✅ Classification and notes saved.")
                        st.session_state.saved = True
                        st.session_state.current_notes = ""
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("✅ Prediction and notes already saved. Use 'Try another image' to continue.")

        if st.session_state.results:
            st.markdown("### 📋 Classification History:")
            for entry in reversed(st.session_state.results):
                name = entry[0]
                result = entry[1]
                note = entry[2] if entry[2] else ""
                st.write(f"• {name} → **{result}**")
                if note:
                    st.caption(f"📝 {note}")

            if st.download_button("📤 Export History to CSV", data=pd.DataFrame(st.session_state.results, columns=["Filename", "Prediction", "Notes"]).to_csv(index=False),
                                file_name="classification_history.csv", mime="text/csv"):
                st.success("CSV downloaded!")

        if st.button("🔄 Try another image"):
            st.session_state.step = "ready"
            st.session_state.uploader_key += 1
            st.session_state.classified_once = False
            st.session_state.saved = False
            st.session_state.current_notes = ""
            st.session_state.current_label = ""
            st.session_state.current_filename = ""
            st.rerun()

# GAN section
with col2:
    st.markdown("<div class='section-title'>🧬 Synthetic MRI Generation</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size: 26px;'>Generate a synthetic MRI image based on tumor type using StyleGAN2</p>", unsafe_allow_html=True)

    # Device configuration (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @st.cache_resource
    def load_stylegan_model_local(local_path):
        with open(local_path, "rb") as f:
            G = pickle.load(f)['G_ema'].to(device)
        return G

    def generate_image(G):
        z = torch.randn([1, G.z_dim]).to(device)
        label = torch.zeros([1, G.c_dim]).to(device)
        img = G(z, label, truncation_psi=0.5, noise_mode='const')[0]
        img = (img.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

    @st.cache_resource
    def load_model_paths():
        return {
            "Glioma": hf_hub_download(
                repo_id="SergioJaraa/MRI_Tumor_StyleGAN2",
                filename="network-snapshot-000480.pkl",
                repo_type="model"
            ),
            "Meningioma": hf_hub_download(
                repo_id="SergioJaraa/MRI_Tumor_StyleGAN2",
                filename="meningioma_model.pkl",
                repo_type="model"
            ),
            "Pituitary": hf_hub_download(
                repo_id="SergioJaraa/MRI_Tumor_StyleGAN2",
                filename="pituitary_model.pkl",
                repo_type="model"
            ),
        }

    models = load_model_paths()


    model_name = st.selectbox("Select model to generate image", list(models.keys()))

    if st.button("🎲 Generate Image"):
        with st.spinner("Generating image..."):
            G = load_stylegan_model_local(models[model_name])
            img = generate_image(G)

            st.image(img, caption=f"Generated by {model_name}", use_column_width=True)

            # Save temporarily for download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name)
                with open(tmp.name, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Image",
                        data=file,
                        file_name=f"{model_name.lower().replace(' ', '_')}_generated.png",
                        mime="image/png",
                        use_container_width=True
                    )

        # Optional: Allow regenerating new image
        if st.button("🔁 Generate Another Image"):
            st.rerun()