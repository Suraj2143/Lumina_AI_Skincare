# ✨ Lumina Skincare AI

Lumina is a full-stack AI application that combines Computer Vision, Recommendation Systems, and Generative AI (RAG) to provide personalized skincare recommendations and dermatology advice.

## 🚀 The Problem
Standard skincare recommendation engines rely on manual user input (asking the user to know their own skin type), which creates a poor user experience and leads to bad recommendations. 

## 💡 The Solution
I built an end-to-end pipeline that analyzes a user's photo, explains *why* it made its decision, and acts as a conversational AI Dermatologist.

### Key Features:
*   **🧠 Custom Vision Model:** A fine-tuned PyTorch ResNet18 model that classifies skin types (Dry vs. Oily) directly from user-uploaded facial photos.
*   **👁️ Explainable AI (XAI):** Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize exactly which parts of the face the neural network is focusing on (e.g., the T-zone). This builds user trust.
*   **🛡️ Fail-Safe Inference:** Implemented a 75% Softmax confidence threshold. If the model is unsure (e.g., a blurry photo), it gracefully falls back to manual user input to prevent bad recommendations.
*   **⚙️ Hybrid Recommendation Engine:** Combined rule-based skin-type filtering with a custom feature-engineering scoring system. It boosts products containing specific active ingredients (e.g., Salicylic Acid for Acne) based on user-selected concerns.
*   **💬 LLM Dermatologist (RAG):** A Retrieval-Augmented Generation pipeline powered by Llama 3 (via Groq API). It queries the local product database and injects the retrieved context into the LLM to answer user questions accurately, eliminating hallucinations.

## 🛠️ Tech Stack
*   **Frontend:** Streamlit
*   **Deep Learning:** PyTorch, Torchvision
*   **Explainability:** pytorch-grad-cam, OpenCV
*   **Generative AI:** Groq API (Llama 3)
*   **Data Processing:** Pandas, NumPy, Scikit-learn

## 📦 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Suraj2143/Lumina_AI_Skincare.git
   cd Lumina_AI_Skincare