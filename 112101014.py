import json
import os
from tqdm import tqdm
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from unsloth import FastLanguageModel

# Load and preprocess data
def load_train_data(train_path):
    with open(train_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def load_test_data(test_path):
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    return test_data

"""def prepare_dataset(data):
    return Dataset.from_dict({
        "input_text": [item['introduction'] for item in data],
        "target_text": [item['abstract'] for item in data]
    })"""

def prepare_dataset(data):
    prompts = [
        f"以下是論文介紹：{item['introduction']} 摘要為：{item['abstract']}"
        for item in data
    ]
    return Dataset.from_dict({"text": prompts})

# Tokenize function
"""def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], max_length=1024, padding="max_length", truncation=True
    )
    labels = tokenizer(
        examples["target_text"], max_length=256, padding="max_length", truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs"""

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

# Load model and tokenizer
# model_name = "facebook/bart-base"
# model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)
"""tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    # task_type="CAUSAL_LM"
)

# Load and tokenize training data
train_data = load_train_data("train.json")
# train_dataset = prepare_dataset(train_data)
train_dataset = prepare_dataset(train_data)
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# Define training arguments
"""training_args = Seq2SeqTrainingArguments(
    output_dir="./bart_abstract_gen",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    report_to="none"
)"""

training_args = TrainingArguments(
    output_dir="./llama_lora_abstract_gen",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    fp16=True,
)

# Trainer
"""trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)"""

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train
trainer.train()

# Load test set
test_data = load_test_data("test.json")

# Predict abstracts
results = []
"""for item in tqdm(test_data):
    inputs = tokenizer(item["introduction"], return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()} if torch.cuda.is_available() else inputs
    summary_ids = model.generate(**inputs, max_length=256, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    results.append({"paper_id": item["paper_id"], "abstract": summary})"""

"""for item in tqdm(test_data):
    prompt = f"以下是論文介紹：{item['introduction']} 摘要為："
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.cuda() for k, v in inputs.items()} if torch.cuda.is_available() else inputs

    outputs = model.generate(
        **inputs,
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )"""

for item in tqdm(test_data):
    prompt = f"以下是論文介紹：{item['introduction']} 摘要為："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({"paper_id": item["paper_id"], "abstract": summary.split("摘要為：")[-1].strip()})

    """summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    abstract_only = summary.split("摘要為：")[-1].strip()
    results.append({"paper_id": item["paper_id"], "abstract": abstract_only})"""

# Save to submission file
with open("your_student_id.json", "w") as f:
    for r in results:
        json.dump(r, f)
        f.write("\n")

print("✅ Done. Submission saved as your_student_id.json.")