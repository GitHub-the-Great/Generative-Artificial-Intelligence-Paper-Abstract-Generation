# Generative-Artificial-Intelligence-Paper-Abstract-Generation
# HW2 Paper Abstract Generation

This repository contains my solution for HW2: **generate paper abstracts from paper introduction bodies**. The approach fine-tunes a lightweight LLM with LoRA and then generates abstracts for the test introductions. :contentReference[oaicite:0]{index=0}

---

## Task Summary

- **Input**: paper **introductions**
- **Output**: generated **abstracts**
- **Training data**: `train.json` (JSON Lines)
- **Test data**: `test.json` (JSON Lines)
- **Submission format**: `{student_id}.json` following `sample_submission.json` style (one JSON object per line). :contentReference[oaicite:1]{index=1}

---

## Model & Method

- **Base model**: `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- **Tuning**: LoRA with Unsloth + HuggingFace Trainer
- **Objective**: Causal LM style prompt-to-target learning
- **Prompt template (training/inference)**:
  - `以下是論文介紹：<introduction> 摘要為：` :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Expected Directory Structure (for E3 submission)

After unzipping `{student_id}.zip`, the folder should include: :contentReference[oaicite:4]{index=4}

```

{student_id}.zip
└── {student_id}_code/        # code to reproduce results
└── Model_checkpoint/         # checkpoint for reproducing test output
└── requirements.txt
└── readme.md
└── {student_id}.json         # final predictions

````

> Do **NOT** include datasets in the zip. Please clearly state where `train.json` and `test.json` should be placed. :contentReference[oaicite:5]{index=5}

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
````

If `unsloth` is not available from PyPI in your environment:

```bash
pip install git+https://github.com/unslothai/unsloth.git
```

`requirements.txt` includes at least:

* torch
* transformers
* datasets
* tqdm
* unsloth 

---

## Data Preparation

Place the datasets in the **same directory** as the script (or adjust paths in code):

```
train.json
test.json
```

Each line in:

* `train.json` has: `paper_id`, `introduction`, `abstract`
* `test.json` has: `paper_id`, `introduction` 

---

## Reproducing My Results

From the project root:

```bash
python 112101014.py
```

The script will:

1. Load `train.json`
2. Construct instruction-style prompts
3. Fine-tune the base model with LoRA
4. Load `test.json`
5. Generate abstracts
6. Save predictions to an output JSON Lines file

---

## Output

The final prediction file should be named:

```
112101014.json
```

(or equivalently `{student_id}.json` for submission)

Format (one object per line):

```json
{"paper_id": 0, "abstract": "..." }
```

Example file is provided in this package. 

---

## Hardware Notes

* A CUDA GPU is recommended for practical training/inference speed.
* You can reduce memory usage by lowering batch size or generation length. 

---

## Notes on Reproducibility

* This code is designed to reproduce the final `{student_id}.json` using the included checkpoint.
* Dataset files are expected to be provided by the TA/E3 environment and placed as described above. 

```
```
