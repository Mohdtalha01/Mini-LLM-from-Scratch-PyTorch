🧠 Mini LLM from Scratch (PyTorch)

A from-scratch implementation of a modern Transformer-based Language Model built using PyTorch. This project demonstrates core concepts behind Large Language Models (LLMs) including normalization, attention optimizations, and efficient training techniques.

🚀 Features
✅ Built a Transformer Language Model from scratch
✅ Implemented modern components:
RMSNorm
SwiGLU activation
RoPE (Rotary Positional Embeddings)
GQA (Grouped Query Attention)
✅ Character-level tokenizer
✅ Trained on Tiny Shakespeare dataset
✅ GPU/CPU compatible training
✅ Text generation with sampling
📂 Project Structure
├── code.ipynb        # Main notebook (model + training + generation)
├── shakespeare.txt   # Dataset (auto-downloaded)
└── README.md         # Project documentation
⚙️ Setup & Installation
1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
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
Device Support:
CPU
GPU (CUDA)
Apple Silicon (MPS)
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
Switch to subword tokenization (BPE)
Add evaluation metrics (perplexity)
Convert into a deployable API
👨‍💻 Author

Mohd Talha
M.Tech (AI/ML) — Batch Topper

⭐ Acknowledgements
Inspired by modern LLM architectures
Dataset: Tiny Shakespeare (by Andrej Karpathy)
