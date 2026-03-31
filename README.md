A from-scratch implementation of a modern **Transformer-based Language Model** built using **PyTorch**. This project demonstrates core concepts behind Large Language Models (LLMs), including normalization techniques, attention optimizations, and efficient training strategies.

---

## 🚀 Features

- ✅ Built a **Transformer Language Model from scratch**
- ✅ Implemented modern components:
  - **RMSNorm**
  - **SwiGLU activation**
  - **RoPE (Rotary Positional Embeddings)**
  - **GQA (Grouped Query Attention)**
- ✅ Character-level tokenizer
- ✅ Trained on **Tiny Shakespeare dataset**
- ✅ Compatible with **CPU, GPU (CUDA), and Apple Silicon (MPS)**
- ✅ Text generation using **temperature sampling**

---

## 📂 Project Structure


├── code.ipynb # Main notebook (model, training, generation)
├── shakespeare.txt # Dataset (auto-downloaded)
└── README.md # Project documentation


---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mini-llm-from-scratch.git
cd mini-llm-from-scratch
2. Install dependencies
pip install torch matplotlib
▶️ Running the Project

Run the Jupyter Notebook:

jupyter notebook code.ipynb

Or open directly in Google Colab.

📊 Model Details
Architecture: Transformer (Decoder-only)
Tokenization: Character-level
Dataset: Tiny Shakespeare
Training Objective: Next token prediction (Language Modeling)
🧪 Training Pipeline
Load and preprocess text data
Build vocabulary and tokenizer
Define Transformer architecture
Train using cross-entropy loss
Generate text using temperature sampling
✨ Sample Output
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer...

(Generated after training)

📚 Concepts Covered
Transformer Architecture
Self-Attention Mechanism
Positional Encoding (RoPE)
Normalization (RMSNorm)
Efficient Attention (GQA)
Language Modeling
🎯 Future Improvements
Add KV Cache optimization
Train on larger datasets
Implement subword tokenization (BPE)
Add evaluation metrics (e.g., perplexity)
Convert into a deployable API
👨‍💻 Author

Mohd Talha
M.Tech (AI/ML) — Batch Topper

⭐ Acknowledgements
Inspired by modern LLM architectures
Dataset: Tiny Shakespeare (by Andrej Karpathy)
