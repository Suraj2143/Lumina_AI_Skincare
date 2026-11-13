# ✨ GlowMatch AI: BeautyTech Recommendation & Dermatology Assistant

GlowMatch is an AI-powered, full-stack BeautyTech application that combines Computer Vision, Recommendation Systems, and Generative AI (RAG) to provide personalized skincare recommendations.

## 🚀 Key Features
*   **🧠 Custom Deep Learning Model:** A fine-tuned PyTorch ResNet18 model that classifies skin types (Dry vs. Oily) directly from user-uploaded facial photos.
*   **👁️ Explainable AI (XAI):** Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize exactly which parts of the face the neural network is focusing on (e.g., the T-zone).
*   **🛡️ Fail-Safe Inference:** Implements a 75% Softmax confidence threshold. If the model is unsure, it gracefully falls back to manual user input to prevent bad recommendations.
*   **⚙️ Hybrid Recommendation Engine:** Combines rule-based skin-type filtering with a custom feature-engineering scoring system that boosts products containing specific active ingredients (e.g., Salicylic Acid for Acne).
*   **💬 LLM Dermatologist (RAG):** A Retrieval-Augmented Generation pipeline powered by Llama 3 (via Groq API). It queries the product database and uses the retrieved context to answer user questions accurately, eliminating hallucinations.

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **Deep Learning:** PyTorch, Torchvision
*   **Explainability:** pytorch-grad-cam, OpenCV
*   **Generative AI:** Groq API (Llama 3)
*   **Data Processing:** Pandas, NumPy, Scikit-learn

## 📦 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/glowmatch-beautytech-recommender.git
   cd glowmatch-beautytech-recommender