"""
Lumina Skincare AI — Luxury BeautyTech Recommendation System with RAG
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import sys, os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import cv2
from groq import Groq

# Import Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from recommender import LuminaRecommender, SKIN_TYPES

st.set_page_config(page_title="Lumina | AI", page_icon="✨", layout="wide")

# ----------------------------------------------------------------------
# GROQ LLM SETUP
# ----------------------------------------------------------------------
# Get your free key at console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") # Reads from .env file
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------------------------------------------------------------
# AI MODEL LOADING & CONFIDENCE
# ----------------------------------------------------------------------
@st.cache_resource
def load_skin_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2) 
    try:
        model.load_state_dict(torch.load("skin_model.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        return None

ai_model = load_skin_model()
ai_skin_types = ["Dry", "Oily"] 

# ----------------------------------------------------------------------
# CURRENCY
# ----------------------------------------------------------------------
USD_TO_INR = 83
def to_inr(usd_amount): return round(usd_amount * USD_TO_INR)
def format_inr(amount):
    n = int(amount); s = str(n)
    if len(s) <= 3: return s
    last3, rest = s[-3:], s[:-3]; parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:]); rest = rest[:-2]
    if rest: parts.insert(0, rest)
    return ",".join(parts) + "," + last3

# ----------------------------------------------------------------------
# THEME SYSTEM (Lumina Theme)
# ----------------------------------------------------------------------
THEMES = {
    "light": {
        "bg1": "#F7E9E4", "bg2": "#F0EAE0", "base": "#FBF6F1",
        "card": "#FFFFFF", "card_border": "#EFE5DA", "card_border_hover": "#E3D2C4",
        "ink": "#2B2320", "muted": "#6B5D53", "muted2": "#9C8C7E",
        "wine": "#6E2A3E", "wine_hover": "#55202F", "wine_light": "#FBF0F2",
        "sidebar": "#F5EDE4", "border": "#E8DDD3", "track": "#F0E6DB",
        "shadow": "rgba(110,42,62,0.14)", "badge_text": "#FFFFFF",
    },
    "dark": {
        "bg1": "#241A1E", "bg2": "#1D1614", "base": "#181211",
        "card": "#241C1A", "card_border": "#3A2E29", "card_border_hover": "#54423A",
        "ink": "#F3E9E1", "muted": "#C4B3A6", "muted2": "#9C8C7E",
        "wine": "#E8A0B4", "wine_hover": "#F0B7C6", "wine_light": "#3A222A",
        "sidebar": "#1F1715", "border": "#3A2E29", "track": "#3A2E29",
        "shadow": "rgba(232,160,180,0.12)", "badge_text": "#1D1614",
    },
}
CATEGORY_COLORS = {"Moisturizer": "#2E7D6B", "Cleanser": "#3B6FA0", "Face Mask": "#B8467A", "Treatment": "#B08A1E", "Eye cream": "#7B5EA7", "Sun protect": "#C9772E"}
CATEGORY_ICONS = {"Moisturizer": "💧", "Cleanser": "🫧", "Face Mask": "🎭", "Treatment": "✨", "Eye cream": "👁", "Sun protect": "☀️"}

if "dark_mode" not in st.session_state: st.session_state["dark_mode"] = False
if "favorites" not in st.session_state: st.session_state["favorites"] = set()

T = THEMES["dark"] if st.session_state["dark_mode"] else THEMES["light"]

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, .stApp {{
        background: radial-gradient(1200px 500px at 15% -10%, {T['bg1']} 0%, {T['base']} 45%),
                    radial-gradient(900px 500px at 100% 0%, {T['bg2']} 0%, {T['base']} 55%);
        font-family: 'Inter', sans-serif;
    }}
    * {{ color: {T['ink']}; }}
    .gm-cat-tag, .gm-cat-tag * {{ color: {T['badge_text']} !important; }}
    div.stButton > button, div.stButton > button * {{ color: #FFFFFF !important; }}
    
    .gm-hero {{ padding: 8px 0 20px 0; border-bottom: 1px solid {T['border']}; margin-bottom: 24px; }}
    .gm-eyebrow {{ font-size: 12px; letter-spacing: 2px; text-transform: uppercase; color: {T['wine']}; font-weight: 700; margin-bottom: 6px; }}
    .gm-title {{ font-family: 'Fraunces', serif; font-size: 46px; font-weight: 700; margin: 0; line-height: 1.05; background: linear-gradient(100deg, {T['ink']} 45%, {T['wine']} 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }}
    .gm-sub {{ font-size: 15px; color: {T['muted']}; margin-top: 10px; max-width: 640px; line-height: 1.5; }}
    
    .gm-card {{ background: {T['card']}; border-radius: 14px; padding: 16px 20px; margin-bottom: 16px; border: 1px solid {T['card_border']}; position: relative; height: 100%; }}
    .gm-card:hover {{ box-shadow: 0 10px 28px {T['shadow']}; transform: translateY(-3px); border-color: {T['card_border_hover']}; }}
    .gm-name {{ font-family: 'Fraunces', serif; font-size: 17px; font-weight: 600; line-height: 1.25; margin-bottom: 2px; min-height: 44px; }}
    .gm-brand {{ font-size: 12.5px; color: {T['muted2']}; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; }}
    .gm-price {{ font-family: 'Fraunces', serif; font-size: 19px; font-weight: 700; color: {T['wine']}; }}
    
    section[data-testid="stSidebar"] {{ background-color: {T['sidebar']}; border-right: 1px solid {T['border']}; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading...")
def load_engine(): return LuminaRecommender()
rec = load_engine()

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    top = st.columns([3, 1])
    with top[0]: st.markdown("### 💧 Your skin profile")
    with top[1]: st.toggle("🌙", key="dark_mode", help="Dark mode")

    uploaded_file = st.file_uploader("Upload a clear facial photo", type=["jpg", "jpeg", "png"])
    
    detected_skin_type = None
    if uploaded_file is not None and ai_model is not None:
        image = Image.open(uploaded_file).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = ai_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            detected_skin_type = ai_skin_types[predicted.item()]
            conf_score = confidence.item() * 100

        if conf_score > 75:
            st.success(f"AI detected: **{detected_skin_type} Skin** (Confidence: {conf_score:.1f}%)")
            
            rgb_img = np.float32(image) / 255.0
            cam = GradCAM(model=ai_model, target_layers=[ai_model.layer4[-1]])
            targets = [ClassifierOutputTarget(predicted.item())]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            col1, col2 = st.columns(2)
            with col1: st.image(image, caption="Original", use_container_width=True)
            with col2: st.image(visualization, caption="AI Focus", use_container_width=True)
        else:
            st.warning(f"AI is unsure ({conf_score:.1f}%). Please select manually.")
            detected_skin_type = None

    default_index = SKIN_TYPES.index(detected_skin_type) if detected_skin_type in SKIN_TYPES else 0
    skin_type = st.selectbox("Confirm Skin Type", SKIN_TYPES, index=default_index)
    
    concerns = st.multiselect("Target Concerns", ["Acne", "Aging", "Dryness", "Redness"])
    category_filter = st.selectbox("Category", ["All"] + sorted(rec.products["Label"].unique().tolist()))
    max_price_inr = st.slider("Budget (max ₹)", 0, 35000, 15000, step=500)

# ----------------------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="gm-hero">
    <div class="gm-eyebrow">BeautyTech recommendation engine</div>
    <div class="gm-title">Find your match, by formula</div>
    <div class="gm-sub">Hybrid recommendations built on real ingredient science, semantic vector search, and LLM-powered explanations.</div>
</div>
""", unsafe_allow_html=True)

def apply_concern_scoring(df, selected_concerns):
    if not selected_concerns or df.empty: return df
    col_name = next((col for col in df.columns if col.lower() == "ingredients"), None)
    if col_name is None: return df
    concern_ing = {"Acne": ["salicylic acid", "niacinamide", "zinc"], "Aging": ["retinol", "peptide", "vitamin c"], "Dryness": ["hyaluronic acid", "ceramide"], "Redness": ["centella", "cica", "aloe"]}
    targets = []
    for c in selected_concerns: targets.extend(concern_ing.get(c, []))
    df = df.copy()
    df["concern_match"] = df[col_name].apply(lambda x: sum(1 for t in targets if t in str(x).lower()))
    return df.sort_values(by=["concern_match", "Rank"], ascending=[False, False])

def render_grid(df, key_prefix="grid"):
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.iterrows()):
        color = CATEGORY_COLORS.get(row["Label"], T["wine"])
        icon = CATEGORY_ICONS.get(row["Label"], "•")
        pid = int(row["product_id"])
        with cols[i % 3]:
            st.markdown(f"""
            <div class="gm-card">
                <span class="gm-cat-tag" style="background:{color}; color:#FFF; padding:3px 10px; border-radius:20px; font-size:10px;">{icon} {row['Label']}</span>
                <div class="gm-name">{row['Name']}</div>
                <div class="gm-brand">{row['Brand']}</div>
                <div class="gm-price">₹{format_inr(to_inr(row['Price']))}</div>
            </div>
            """, unsafe_allow_html=True)

# --- TABS ---
tab_labels = ["✨ For You", "💬 Ask AI Dermatologist"]
tabs = st.tabs(tab_labels)

with tabs[0]:
    st.subheader(f"Picked for {skin_type.lower()} skin")
    results = rec.recommend_for_skin_profile(skin_type, category=None if category_filter == "All" else category_filter, max_price=max_price_inr / USD_TO_INR, n=30)
    results = apply_concern_scoring(results, concerns).head(9)
    render_grid(results, key_prefix="foryou")

# --- NEW TAB: RAG LLM CHATBOT ---
with tabs[1]:
    st.subheader("Ask the AI Dermatologist")
    st.caption("Powered by Llama 3 via Groq. The AI reads the actual products to answer your questions.")
    
    if client is None or GROQ_API_KEY == "":
        st.error("Please set your GROQ_API_KEY in the .env file to use this feature.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., What is the best moisturizer for oily skin?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing products..."):
                    
                    # Grab the top 3 products for their skin type
                    matched_products = rec.recommend_for_skin_profile(skin_type, n=3)
                    
                    # Build the context
                    context = "Here are some products from our catalog:\n"
                    for _, p in matched_products.iterrows():
                        context += f"- {p['Name']} by {p['Brand']} (Category: {p['Label']}, Price: ₹{format_inr(to_inr(p['Price']))}).\n"

                    system_prompt = f"""You are Lumina, an expert AI dermatologist. 
                    The user has {skin_type.lower()} skin. 
                    Use the following product context to answer their question. 
                    Be professional, concise, and helpful.
                    Context: {context}"""

                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            model="openai/gpt-oss-20b",
                        )
                        response = chat_completion.choices[0].message.content
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"LLM Error: {e}")