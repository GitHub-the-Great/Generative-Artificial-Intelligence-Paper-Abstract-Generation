Abstract Generator using LLaMA-LoRA (Unsloth) and HuggingFace Trainer
This project fine-tunes a language model (LLaMA-3.1 8B with LoRA) on a dataset of paper introductions and abstracts, and generates abstracts for unseen introductions. It uses the Unsloth framework and HuggingFace's transformers library.

📁 Directory Structure
.
├── Model_checkpoint          
├── 112101014_code            # Main training and inference script
├── 112101014.json      # Final output predictions
├── requirements.txt          # Python dependencies
└── README.md                 # You're reading this!
🔧 Setup
Install dependencies:
pip install -r requirements.txt
(Optional) Install Unsloth from GitHub if needed:
pip install git+https://github.com/unslothai/unsloth.git
Ensure you have the following files ready:

train.json: Each line contains a JSON object with "introduction" and "abstract".

test.json: Each line contains a JSON object with "paper_id" and "introduction".

🚀 Run the Code
Simply execute the Python script:
python 112101014.py
This script will:

Load and preprocess the training data

Fine-tune the LLaMA-3.1 8B model with LoRA via Unsloth

Tokenize and train using HuggingFace Trainer

Load test.json, generate abstracts for each introduction

Save the generated results to your_student_id.json

📄 Output Format
The final file your_student_id.json contains one JSON object per line:
{"paper_id": "abc123", "abstract": "這是一篇關於深度學習的研究摘要..."}
🧠 Model
Base Model: unsloth/Meta-Llama-3.1-8B-bnb-4bit

Training: LoRA fine-tuning with Trainer (fp16)

Prompt Format:

以下是論文介紹：<introduction> 摘要為：
✅ Tips
Use a GPU with ≥16GB VRAM (or Colab Pro) for smooth training/inference.

Adjust per_device_train_batch_size or max_new_tokens as needed for memory constraints.
