import streamlit as st
from transformers import pipeline, set_seed

st.set_page_config(page_title="Text Generator", layout="centered", page_icon="🧠")

st.markdown("""
    <style>
        body {
            background-color: #fdf5e6;
        }
        h1, h2, h3, h4, h5, h6 {
        color: #FF4B4B;}
        .main {
            background-color: #fdf5e6;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 GENERATIVE TEXT MODEL")

model_type = st.selectbox("Select Model Type", ["GPT-2", "LSTM (Simulated)"])

prompt = st.text_area("Enter your topic prompt:", placeholder="e.g., The impact of AI on society")

if st.button("Generate Text"):
    if not prompt.strip():
        st.warning("Please enter a valid prompt.")
    else:
        st.info(f"Generating text using **{model_type}**...")
        set_seed(42)

        if model_type == "GPT-2":
            @st.cache_resource
            def load_gpt_model():
                return pipeline('text-generation', model='gpt2')
            
            generator = load_gpt_model()
            output = generator(prompt, max_new_tokens=100, num_return_sequences=1)
            st.success(output[0]['generated_text'])

        elif model_type == "LSTM (Simulated)":
            simulated_text = f"{prompt.strip()}... (This is a simulated LSTM response. Real LSTM generation would require training a model.)"
            st.success(simulated_text)
