# How to Run

## 1. Create Virtual Environment

```bash
python3 -m venv .venv
```

## 2. Activate Environment

### Linux / Mac

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Requirements

```bash
pip install -r requirements-torch-cu128.txt
pip install -r requirements.txt
```

---


## 6. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=llm-env
```

---

## 7. Verify GPU Access

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```python
True
```

---

## 8. Start Jupyter

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

---

# Supported Models

This environment supports fine-tuning and supervised fine-tuning (SFT) for:

- Qwen
- Mistral
- DeepSeek
- SmolLM
- LLaMA-family models
- Gemma
- Phi
- Other Hugging Face Transformer models

Supports:

- Full fine-tuning
- LoRA
- QLoRA
- 4-bit quantization
- Instruction tuning
- Continued pretraining
- Multi-GPU training
- Evaluation pipelines