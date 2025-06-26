import streamlit as st
from PIL import Image
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tempfile
import torch
import pickle
import numpy as np
from evaluate import load_model, preprocess_image, classify_image
import torch.nn.functional as F
import pandas as pd
from huggingface_hub import hf_hub_download
import urllib.request

# Fix sys.path so pickle.load finds the right modules
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "stylegan2_ada_pytorch"))
torch_utils_path = os.path.join(base_path, "torch_utils")
training_path = os.path.join(base_path, "training")
for p in [base_path, torch_utils_path, training_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stylegan2_ada_pytorch.training import networks


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
            font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
            background-color: #0d1117;
            color: #d1d5db;
            margin: 0;
            padding: 0;
        }
        img {
            max-width: 500px ;
            height: auto ; 
            display: block ; 
            margin-left: auto ; 
            margin-right: auto;
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
            color: #e0e7ff;
            font-family: 'Courier New', Courier, monospace;
            background: linear-gradient(90deg, #58a6ff, #d1d5db);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            padding: 10px;
            border-radius: 8px;
        }

        .section-wrapper {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: center;
            height: 100%;
            padding-bottom: 20px;
        }

        .section-title {
            margin-bottom: 10px;
            font-size: 32px;
            font-weight: 600;
            color: #58a6ff;
            text-align: center;
            font-family: 'Courier New', Courier, monospace;
        }

        .section-text {
            margin-top: 0;
            font-size: 26px;
            text-align: center;
        }

        .block-container {
            padding: 2rem 2rem;
            max-width: 2200px;
            margin: 0 auto;
        }

        button {
            background-color: #1f2a44;
            color: #e0e7ff;
            border: 1px solid #58a6ff;
            padding: 0.7em 1.5em;
            border-radius: 10px;
            font-size: 1.9rem;
            font-weight: 500;
            transition: all 0.3s ease-in-out;
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            box-shadow: 0 4px 6px rgba(88, 166, 255, 0.1);
        }

        button:hover {
            background-color: #58a6ff;
            color: #e0e7ff;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(88, 166, 255, 0.2);
        }

        .stButton > button {
            display: block;
            margin: 0 auto;
            width: auto !important;
            font-size: 1.9rem;
        }

        section[data-testid="stFileUploader"] {
            background-color: #161b22;
            border: 2px solid #2d333b;
            padding: 1.5em;
            border-radius: 12px;
            transition: border-color 0.3s ease;
        }

        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: center; /* Centers content horizontally */
            justify-content: flex-start;
            padding-bottom: 50px;
            padding-top: 50px;
            background-color: #161b22;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(88, 166, 255, 0.1);
            transition: transform 0.3s ease;
            padding-right: 50px;
            padding-left: 50px;
            margin-top: 20px;
        }

        section[data-testid="stFileUploader"]:hover {
            border-color: #58a6ff;
        }

        div[data-baseweb="select"] {
            background-color: #161b22 !important;
            border-radius: 8px;
            border: 2px solid #2d333b;
            transition: border-color 0.3s ease;
        }

        div[data-baseweb="select"]:hover {
            border-color: #58a6ff;
        }

        div[data-baseweb="select"] > div {
            color: #e0e7ff;
        }

        textarea {
            background-color: #161b22 !important;
            color: #e0e7ff !important;
            border-radius: 8px !important;
            border: 2px solid #2d333b !important;
            padding: 0.8em;
            transition: border-color 0.3s ease;
        }

        textarea:focus {
            border-color: #58a6ff !important;
            outline: none;
        }

        .result {
            background-color: #1c2128;
            border: 2px solid #58a6ff;
            padding: 1.5em;
            border-radius: 12px;
            text-align: center;
            color: #e0e7ff;
            font-size: 24px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(88, 166, 255, 0.1);
        }

        .stRadio > div {
            justify-content: center;
            gap: 1.5em;
        }

        .st-expander {
            background-color: #161b22 !important;
            border: 2px solid #2d333b !important;
            border-radius: 10px;
            transition: border-color 0.3s ease;
            padding: 20px;
            font-size: 3rem;
        }

        .st-expander:hover {
            border-color: #58a6ff !important;
        }

        .stDownloadButton {
            margin-top: 1em;
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
col1, col2 = st.columns([1.5, 1.5])

# Classify section
with col1:
    st.markdown("""
        <div class="section-wrapper">
            <div class='section-title'>🔬 Classify Tumor Image</div>
            <div class='section-text'>
                Test your knowledge trying to guess the tumor and check with the Federated Learning CNN Model trained by three clients
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 28px; text-align:center; margin-bottom: 10px; padding-top: 20px'>Upload image of MRI brain tumor</div>", unsafe_allow_html=True)

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload brain MRI", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key=f"file_uploader_{st.session_state.uploader_key}")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("L")
            st.markdown("<div style='text-align: center; display: flex; justify-content: center;'>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", width=1000)
            st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("""
        <div class="section-wrapper">
            <div class='section-title'>🧬 Synthetic MRI Generation</div>
            <div class='section-text'>Generate a synthetic MRI image based on tumor type using StyleGAN2-ADA-pytorch fine tuned models</div>
        </div>
    """, unsafe_allow_html=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_stylegan_model_local(local_path):
        import sys, os
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "stylegan2_ada_pytorch"))
        torch_utils_path = os.path.join(base_path, "torch_utils")
        if torch_utils_path not in sys.path:
            sys.path.insert(0, torch_utils_path)
        from stylegan2_ada_pytorch.training import networks
        with open(local_path, "rb") as f:
            G = pickle.load(f)['G_ema'].to(device)
        return G

    def generate_image(G):
        z = torch.randn([1, G.z_dim]).to(device)
        img = G(z, None, truncation_psi=0.5, noise_mode='const')[0]
        img = (img.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

    @st.cache_resource
    def load_model_paths():
        return {
            "Glioma": hf_hub_download("SergioJaraa/MRI_Tumor_StyleGAN2", "network-snapshot-000480.pkl", repo_type="model"),
            "Meningioma": hf_hub_download("SergioJaraa/MRI_Tumor_StyleGAN2", "meningioma_model.pkl", repo_type="model"),
            "Pituitary": hf_hub_download("SergioJaraa/MRI_Tumor_StyleGAN2", "pituitary_model.pkl", repo_type="model"),
        }

    models = load_model_paths()

    st.markdown("<div style='font-size: 28px; text-align:center; margin-bottom: 10px; padding-top: 20px'>Select model to generate image</div>", unsafe_allow_html=True)

    model_name = st.selectbox(
    "Select GAN model",
    list(models.keys()),
    key="gan_model_select",
    label_visibility="collapsed"
    )

    if st.button("🎲 Generate Image"):
        with st.spinner("Generating image..."):
            G = load_stylegan_model_local(models[model_name])
            img = generate_image(G)
            st.image(img, caption=f"Generated by {model_name}", width=1000)

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

    if st.button("🔁 Generate Another Image"):
        st.rerun()