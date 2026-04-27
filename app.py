import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import json

# Page Config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f7f0;
    }
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    }
    .title {
        text-align: center;
        color: #2e7d32;
        font-size: 3em;
        font-weight: bold;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #388e3c;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .result-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .healthy {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 10px;
    }
    .disease {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 10px;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .upload-box {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4caf50, #2e7d32);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    footer {
        text-align: center;
        color: #666;
        padding: 20px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Class Names
class_names = [
    'Cabbage_Alternia_Leaf_Spot',
    'Cabbage_Aphid_Colony',
    'Cabbage_Healthy',
    'Cabbage_Ring_Spot',
    'Malabar_Anthracnose',
    'Malabar_Bactorial_Spot',
    'Malabar_Healthy',
    'Malabar_PestDamage',
    'Potato_EarlyBlight',
    'Potato_Healthy',
    'Potato_LaterBlight',
    'Radish_Black_Leaf_Spot',
    'Radish_Flea_Beetle',
    'Radish_Healthy',
    'Radish_Mosaic_Virus'
]

# Load JSON
with open('remedies.json', 'r') as f:
    remedies = json.load(f)

# Crop Quality
crop_quality = {
    'Cabbage_Healthy': '🟢 Excellent — Ready for harvest',
    'Malabar_Healthy': '🟢 Excellent — Ready for harvest',
    'Potato_Healthy': '🟢 Excellent — Ready for harvest',
    'Radish_Healthy': '🟢 Excellent — Ready for harvest',
    'Cabbage_Aphid_Colony': '🟡 Moderate — Needs attention',
    'Malabar_PestDamage': '🟡 Moderate — Monitor closely',
    'Radish_Flea_Beetle': '🟡 Moderate — Monitor closely',
    'Cabbage_Alternia_Leaf_Spot': '🔴 Poor — Treat immediately',
    'Cabbage_Ring_Spot': '🔴 Poor — Treat immediately',
    'Malabar_Anthracnose': '🔴 Poor — Treat immediately',
    'Malabar_Bactorial_Spot': '🔴 Poor — Treat immediately',
    'Potato_EarlyBlight': '🔴 Poor — Treat immediately',
    'Potato_LaterBlight': '🔴 Critical — Remove immediately',
    'Radish_Black_Leaf_Spot': '🔴 Poor — Treat immediately',
    'Radish_Mosaic_Virus': '🔴 Critical — Remove infected plants',
}

# Model Load
@st.cache_resource
def load_model():
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 15)
    model.load_state_dict(torch.load(
        'plant_disease_model.pth',
        map_location=torch.device('cpu')))
    model.eval()
    return model

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Header
st.markdown('<div class="title">🌿 Plant Disease Detection System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Leafy Vegetable Disease Detection & Crop Quality Prediction</div>',
            unsafe_allow_html=True)
st.markdown("---")

# Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div class="metric-card">
            <h2>15</h2>
            <p>Disease Classes</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="metric-card">
            <h2>87%+</h2>
            <p>Model Accuracy</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="metric-card">
            <h2>4</h2>
            <p>Vegetables</p>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
        <div class="metric-card">
            <h2>EfficientNet</h2>
            <p>B3 Model</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Section
left, right = st.columns([1, 1])

with left:
    st.markdown("""
        <div class="upload-box">
            <h3>📸 Upload Leaf Image</h3>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Leaf Image',
                 use_column_width=True)
        detect_btn = st.button('🔍 Detect Disease Now')

with right:
    if uploaded_file is not None and detect_btn:
        with st.spinner('🔬 Analyzing your plant...'):
            model = load_model()
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)

            disease = class_names[predicted.item()]
            conf = confidence.item() * 100
            info = remedies[disease]
            quality = crop_quality[disease]

        st.markdown("### ✅ Analysis Results")

        # Disease Result
        is_healthy = 'Healthy' in disease
        box_class = "healthy" if is_healthy else "disease"
        icon = "✅" if is_healthy else "⚠️"

        st.markdown(f"""
            <div class="{box_class}">
                <h3>{icon} {disease.replace('_', ' ')}</h3>
                <p><b>Confidence:</b> {conf:.1f}%</p>
                <p><b>Crop Quality:</b> {quality}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cause
        st.error(f"🔎 **Cause:** {info['cause']}")

        # Remedy
        st.warning(f"💊 **Remedy:** {info['remedy']}")

        if is_healthy:
            st.balloons()
            st.success("🎉 Great news! Your plant is perfectly healthy!")
        else:
            st.error("⚠️ Please follow the remedy above immediately!")

# Footer
st.markdown("---")
st.markdown("""
    <footer>
        <p>🌿 Plant Disease Detection System | 
        Powered by EfficientNet-B3 & Deep Learning | 
        Major Project 2025</p>
    </footer>
""", unsafe_allow_html=True)