import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from pypdf import PdfReader

# Load the saved model and vectorizer (created by the notebook)
model = joblib.load("news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

STOPWORDS = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


def extract_text_from_file(uploaded_file):
    """Read text out of an uploaded .txt or .pdf file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)
    else:
        # .txt file
        return uploaded_file.read().decode("utf-8", errors="ignore")


st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Keep a running history of checks made in this session
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📰 Fake News Detector")
st.write("Paste a news article below, or upload a .txt / .pdf file, and the model will tell you if it looks Fake or Real.")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file (optional)", type=["txt", "pdf"])

default_text = ""
if uploaded_file is not None:
    try:
        default_text = extract_text_from_file(uploaded_file)
        st.success(f"Loaded text from {uploaded_file.name}. You can edit it below before checking.")
    except Exception as e:
        st.error(f"Could not read that file: {e}")

user_text = st.text_area("Article text (edit if needed):", value=default_text, height=250)

if st.button("Check Article"):
    if user_text.strip() == "":
        st.warning("Please paste some article text first.")
    else:
        cleaned = clean_text(user_text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        fake_pct = probability[0] * 100
        real_pct = probability[1] * 100

        if prediction == 1:
            st.success(f"✅ This looks like REAL news")
        else:
            st.error(f"⚠️ This looks like FAKE news")

        # Confidence bar chart
        st.write("**Confidence breakdown:**")

        st.write(f"Real: {real_pct:.1f}%")
        st.progress(int(real_pct))

        st.write(f"Fake: {fake_pct:.1f}%")
        st.progress(int(fake_pct))

        st.caption(
            "Note: This is a student project model, not a fact-checking tool. "
            "It only recognizes writing patterns similar to its training data."
        )

        # Save this check to the session's history
        snippet = user_text.strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:80] + "..."

        st.session_state.history.insert(0, {
            "Article snippet": snippet,
            "Prediction": "Real" if prediction == 1 else "Fake",
            "Confidence": f"{max(real_pct, fake_pct):.1f}%",
        })

# Show prediction history
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕑 History (this session)")
    st.table(st.session_state.history)

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.caption("Built with Python, Scikit-learn, and Streamlit — News Article Classification project.")
