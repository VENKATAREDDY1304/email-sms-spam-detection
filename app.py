from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
MODEL_NAME_PATH = ROOT / "models" / "best_model_name.txt"

st.set_page_config(page_title="Spam Detection", page_icon="📧", layout="centered")
st.title("📧 Spam Detection System")
st.write("Enter a text message to estimate whether it is spam or legitimate.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Follow the README instructions and run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
model_name = MODEL_NAME_PATH.read_text(encoding="utf-8").strip() if MODEL_NAME_PATH.exists() else "Saved classifier"

message = st.text_area(
    "Message to check",
    placeholder="Example: Congratulations! You have won a prize. Click here to claim...",
    height=150
)

if st.button("Check message", type="primary", use_container_width=True):
    if not message.strip():
        st.error("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)
            spam_index = classes.index("spam") if "spam" in classes else None

        if prediction == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: LEGITIMATE (NOT SPAM)")

        if probabilities is not None and spam_index is not None:
            st.metric("Estimated spam probability", f"{probabilities[spam_index] * 100:.1f}%")
        st.caption(f"Model used: {model_name}")
        st.info("This is a machine-learning estimate, not a guarantee. Avoid clicking suspicious links even if a message is classified as legitimate.")

st.divider()
st.caption("Educational project • NLP + TF-IDF + machine learning")
