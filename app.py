from flask import Flask, request, jsonify, render_template_string, redirect
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import json
import subprocess
import threading
import sys
import torch

# ----------------------------------
# Paths (adjust if needed)
# ----------------------------------
# ----------------------------------
# Paths (adjust if needed)
# ----------------------------------
import os

# ----------------------------------
# Paths (DYNAMIC - Works on Mac)
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
STREAMLIT_SCRIPT = os.path.join(BASE_DIR, "streamlitfrontend.py")

# ----------------------------------
# Load data + models
# ----------------------------------
print("Loading Database and Index...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    ds = json.load(f)

index = faiss.read_index(FAISS_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ### NEW MODEL: LaMini-Flan-T5-783M
# This model is much better at talking naturally than the original Flan-T5.
print("Loading Generative Model (LaMini)...")
gen_model_name = "MBZUAI/LaMini-Flan-T5-783M" 
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
generator = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = generator.to(device)
print(f"Generative Model ready on {device}!")

# Global history for context
conversation_history = []

def retrieve(query, top_k=1):
    query_vec = embedder.encode([query]).astype("float32")
    _, I = index.search(query_vec, top_k)
    return [ds[i] for i in I[0]]

# ----------------------------------
# Flask app
# ----------------------------------
app = Flask(__name__)

streamlit_started = False
streamlit_lock = threading.Lock()

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>FitBot 💪</title>
</head>
<body style="font-family:Arial; text-align:center; padding:50px;">
    <h1>FitBot 💪</h1>
    <p>
        Your AI gym trainer & nutrition assistant.<br>
        Ask about workouts, hypertrophy, muscle groups, bulking, cutting, and food nutrition.
    </p>
    <form action="/launch" method="get">
        <button style="font-size:20px; padding:12px 25px;">
            Chat with FitBot
        </button>
    </form>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HOME_PAGE)

@app.get("/launch")
def launch_streamlit():
    global streamlit_started
    with streamlit_lock:
        if not streamlit_started:
            def run_streamlit():
                if sys.platform == "win32":
                    subprocess.Popen(
                        ["streamlit", "run", STREAMLIT_SCRIPT, "--server.headless", "true"],
                        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    subprocess.Popen(["streamlit", "run", STREAMLIT_SCRIPT, "--server.headless", "true"])
            threading.Thread(target=run_streamlit, daemon=True).start()
            streamlit_started = True
    return redirect("http://localhost:8501")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat():
    global conversation_history
    data = request.json
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    # 1. Retrieve data
    result = retrieve(message, top_k=1)
    retrieved_fact = result[0]["response"]

    # 2. Prepare History (Last 2 turns to keep it focused)
    recent_history = conversation_history[-2:]
    formatted_history = "\n".join(recent_history)

    # ### NEW PROMPT STRUCTURE
    # We ask the model explicitly to "elaborate" so it doesn't give short answers.
    prompt = f"""
    You are a fitness assistant. Answer the user's question using the context provided. Elaborate on the answer.
    
    Context: {retrieved_fact}
    
    History: {formatted_history}
    
    User Question: {message}
    
    Answer:
    """

    # 3. Generate with BETTER Parameters
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    outputs = generator.generate(
        input_ids, 
        max_length=256,          # Increased length (was 150)
        min_length=20,           # Force it to speak at least 20 words
        do_sample=True,
        temperature=0.6,         # Slightly lower temp for coherence
        repetition_penalty=1.2   # Prevents it from getting stuck in loops
    )
    
    natural_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update history
    conversation_history.append(f"User: {message}")
    conversation_history.append(f"Assistant: {natural_response}")

    return jsonify({"response": natural_response})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)