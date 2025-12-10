
# Abstractive Text Summarization with PEFT & LoRA

A modular, production-ready framework for training and evaluating Large Language Models (LLMs) on abstractive summarization tasks. This project supports **T5** and **BART** architectures and implements **Low-Rank Adaptation (LoRA)** for efficient fine-tuning on consumer hardware.

It includes an integrated **Hallucination Verifier** that uses a Natural Language Inference (NLI) model to assess the factual consistency of generated summaries with the source text.

## 🚀 Features

* **Modular Architecture:** Seamlessly switch between models (T5, BART) and datasets (CNN/DailyMail, XSum) using a simple configuration file.
* **Efficient Fine-Tuning:** Implements PEFT (LoRA) and Gradient Checkpointing to reduce memory usage.
* **Hallucination Detection:** Built-in verification pipeline using `roberta-large-mnli` to classify summary sentences as "Faithful" or "Hallucinated".
* **Experiment Tracking:** Fully integrated with **Weights & Biases (W&B)** for real-time logging of loss and metrics.

## 📂 Project Structure
````markdown
```text
Abstractive_Text_Summarization/
├── config.yaml           # Master configuration (Switch models/datasets here)
├── src/
│   ├── data_loader.py    # Handles dataset loading, preprocessing & tokenization
│   ├── model_factory.py  # Loads base models and applies LoRA adapters
│   ├── verifier.py       # NLI-based Hallucination Verifier logic
│   └── utils.py          # Utility functions for config & logging
├── scripts/
│   ├── train.py          # Main training script
│   └── analyze.py        # Inference & Hallucination analysis script
├── requirements.txt      # Project dependencies
└── .env                  # Secrets file (Not uploaded to git)
````

## 🛠️ Installation

### 1\. Clone the repository

```bash
git clone [https://github.com/Seraph-2000/Abstractive_Text_Summarization.git](https://github.com/Seraph-2000/Abstractive_Text_Summarization.git)
cd Abstractive_Text_Summarization
```

### 2\. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

**Step A: Install PyTorch (Hardware Specific)**

  * **For Windows with NVIDIA GPU (Recommended):**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
  * **For Mac / Linux / CPU:**
    ```bash
    pip install torch torchvision torchaudio
    ```

**Step B: Install Project Requirements**

```bash
pip install -r requirements.txt
```

### 4\. Set up Environment Variables

Create a `.env` file in the root directory and add your Weights & Biases API key:

```text
WANDB_API_KEY=your_wandb_api_key_here
```

## 🏃 Usage

### 1\. Configuration

Open `config.yaml` to select your experiment settings. You can uncomment the "Presets" to switch models easily.

```yaml
# Example: Switching to T5-Small on CNN/DailyMail
model_type: "t5"
model_name: "t5-small"
dataset_name: "cnn_dailymail"
```

### 2\. Train the Model

Run the training script. This will download the base model, train the LoRA adapter, and save it to the `models/` directory.

```bash
python scripts/train.py
```

### 3\. Analyze & Verify

Run the analysis script to generate summaries for random samples and verify them against the source text.

```bash
python scripts/analyze.py
```

## 📊 Results & Analysis

The project uses `roberta-large-mnli` to classify summary sentences:

  * 🟢 **[FAITHFUL]**: The sentence is supported by the source document (Entailment).
  * 🔴 **[HALLUCINATION]**: The sentence is unsupported or contradicts the source document.

**Example Output:**

```text
DOCUMENT (Snippet):
The prime minister announced the new budget...
--------------------------------------------------
MODEL SUMMARY:
The prime minister announced the new budget. He also promised free ice cream.
--------------------------------------------------
🔍 Running Hallucination Analysis...
🟢 [FAITHFUL]: The prime minister announced the new budget.
🔴 [HALLUCINATION]: He also promised free ice cream.
```
