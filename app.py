import csv
import pickle
from pathlib import Path

import streamlit as st
import torch

from lm.lstm import LSTMLanguageModel
from lm.generation import generate_text, generate_lstm_text


st.set_page_config(
    page_title="English Language Models",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


@st.cache_resource
def load_models():
    ngram_models = {}

    for name in ["Bigram", "Trigram"]:
        with (ARTIFACT_DIR / f"{name.lower()}.pkl").open("rb") as file:
            ngram_models[name] = pickle.load(file)

    checkpoint = torch.load(
        ARTIFACT_DIR / "lstm_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    vocab_list = checkpoint["vocab_list"]

    assert all(
        saved["vocab_list"] == vocab_list
        for saved in ngram_models.values()
    )

    token_to_id = {
        token: index
        for index, token in enumerate(vocab_list)
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = LSTMLanguageModel(
        **checkpoint["model_config"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return ngram_models, model, vocab_list, token_to_id


ngram_models, lstm_model, vocab_list, token_to_id = load_models()
vocab = set(vocab_list)

st.title("English Language Models")
st.caption(
    "PG-19 alt kümesiyle eğitilmiş "
    "bigram, trigram ve LSTM modellerini deneyin."
)

with st.form("generation_form"):
    model_name = st.selectbox(
        "Model",
        ["Bigram", "Trigram", "LSTM"],
        index=2,
    )

    prompt = st.text_area(
        "İngilizce başlangıç metni",
        value="it was",
        height=100,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        length = st.slider(
            "En fazla yeni token", 1, 100, 30
        )

    with col2:
        top_k = st.slider(
            "Top-k", 1, 50, 10
        )

    with col3:
        seed = st.number_input(
            "Seed", min_value=0, value=42, step=1
        )

    submitted = st.form_submit_button(
        "Metin üret", type="primary"
    )

if submitted:
    if not prompt.strip():
        st.warning("Bir başlangıç metni gir.")
    else:
        with st.spinner("Metin üretiliyor..."):
            if model_name in ngram_models:
                saved = ngram_models[model_name]

                text = generate_text(
                    prompt=prompt,
                    n=saved["n"],
                    ngram_counts=saved["ngram_counts"],
                    context_counts=saved["context_counts"],
                    vocab=vocab,
                    max_new_tokens=length,
                    k=saved["k"],
                    top_k=top_k,
                    seed=int(seed),
                )
            else:
                text = generate_lstm_text(
                    model=lstm_model,
                    prompt=prompt,
                    token_to_id=token_to_id,
                    id_to_token=vocab_list,
                    max_new_tokens=length,
                    temperature=1.0,
                    top_k=top_k,
                    seed=int(seed),
                )

        st.session_state["generated_text"] = text

if "generated_text" in st.session_state:
    st.text_area(
        "Üretilen metin",
        value=st.session_state["generated_text"],
        height=220,
        disabled=True,
    )

results_path = PROJECT_DIR / "results" / "test_results.csv"

if results_path.exists():
    with st.expander("Test karşılaştırması"):
        with results_path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        st.table([
            {
                "Model": row["model"],
                "Test perplexity": round(
                    float(row["test_perplexity"]), 2
                ),
            }
            for row in rows
        ])