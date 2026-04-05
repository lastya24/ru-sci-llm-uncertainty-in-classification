# ru-sci-llm-uncertainty-in-classification

authors: Anisimova A., Limonova A., Shatskikh A.

**Uncertainty-aware classification of Russian scientific texts and user intents using Qwen2.5-1.5B-Instruct.**  
We implement and compare three uncertainty estimation methods from the ACL 2025 survey paper *"A Survey of Uncertainty Estimation Methods on Large Language Models"* (Xia et al.):

- **Verbalized** – model outputs a class and a confidence score (0–1)
- **Self-Consistency** – multiple generations (temperature sampling), uncertainty = agreement / entropy
- **Latent Information** – uses token probabilities (entropy, NLL, margin)

Experiments are conducted on two Russian classification datasets with 10 classes each.

---

## Datasets

| Dataset | Source | Domain | Classes (sample) | Train/Test |
|---------|--------|--------|------------------|-------------|
| **GRNTI** | `ai-forever/ru-scibench-grnti-classification` | Scientific abstracts (GRNTI rubrics) | 10 (e.g., Энергетика, Физика, Биология, Транспорт) | ~1000 / 1000 |
| **CLINC-150 (translated)** | CLINC150 (English) → Russian (translation by LLM) | User intents in a chatbot | 10 (e.g., погода, налоги, рецепт, напоминание) | 90 / 10 (but we used 100 test samples) |

> The GRNTI dataset has **low class separability** (overlapping scientific fields), while CLINC-150 has **high separability** (distinct intents).

---

## Model

**Qwen2.5-1.5B-Instruct** – a lightweight instruction‑tuned LLM. 
We also tested Qwen2.5-0.5B, but it failed to follow the output format reliably.

---

## Methods Implemented

### 1. Verbalized (Single‑round)
The model is prompted to output both the class and a confidence score (0–1) in a strict format.

We then evaluate how well the confidence score distinguishes correct from incorrect predictions (AUROC, AUARC).

### 2. Self‑Consistency
We generate multiple answers (10 per input) with non‑zero temperature (t = 0.5 and t = 1.0).  
Uncertainty is measured as:
- **Confidence** = proportion of generations matching the majority class
- **Certainty** = 1 − (entropy / max_entropy)

Higher agreement → lower uncertainty.

### 3. Latent Information (White‑box)
Using token‑level probabilities from the model, we compute:
- **Entropy** of the next‑token distribution
- **Max probability** of the generated class token
- **Margin** (difference between top‑2 probabilities)
- **Perplexity** (average negative log‑likelihood)

These scores are used as uncertainty measures.

---

## 📈 Results

### Verbalized Method

| Dataset | Accuracy | AUROC (confidence) | AUARC (confidence) | Avg conf (correct) | Avg conf (incorrect) |
|---------|----------|--------------------|--------------------|--------------------|----------------------|
| **GRNTI** (1000 samples) | 0.451 | 0.8076 | 0.6701 | 1.221* | 0.689 |
| **CLINC-150** (100 samples) | 0.860 | 0.8144 | 0.8815 | 0.980 | 0.921 |
| **CLINC-150** (merged "рецепт"/"рецепты") | **0.900** | **0.8644** | **0.9764** | 0.979 | 0.905 |

> *Note: average confidence >1 due to model occasionally outputting values like "0.90" parsed as 0.9 but also "90" parsed as 90 – we later clipped. In the table we show raw numbers from the slide.*

**Key insight:** Verbalized confidence works well when the model can follow instructions (1.5B). On GRNTI (hard task), AUROC is decent (0.81) despite low accuracy – the method still ranks correct answers higher.

---

### Self‑Consistency Method (GRNTI, 1000 samples, 10 generations each)

| Temperature | Metric | AUROC | AUARC |
|-------------|--------|-------|-------|
| **t = 0.5** | Confidence (majority vote) | 0.7551 | 0.7386 |
| **t = 0.5** | Certainty (1 – entropy) | 0.7596 | – |
| **t = 1.0** | Confidence | 0.5719 | 0.4073 |
| **t = 1.0** | Certainty | 0.6116 | – |

> Higher temperature (t=1) makes generations too diverse, hurting uncertainty estimation. t=0.5 works better.

**On CLINC-150 (100 samples, t=1):**  
- AUROC (confidence) = 0.8181  
- AUARC (confidence) = 0.9047  

Self‑consistency performs well on the easier CLINC dataset, especially at t=1.

---

### Latent Information Methods

| Dataset | Metric | Entropy | NLL | Margin |
|---------|--------|---------|----------|--------|
| **GRNTI** | AUROC | 0.815 | 0.808 | 0.771 |
| | AUARC | 0.768 | 0.764 | 0.749 |
| **CLINC-150** | AUROC | 0.947 | 0.956 | 0.932 |
| | AUARC | 0.971 | 0.972 | 0.967 |

**Observation:** Latent information methods (especially negative log likelihood and entropy) achieve near‑perfect uncertainty ranking on the easy CLINC dataset. On GRNTI they are slightly better than verbalized (AUROC 0.82 vs 0.81).

---

## 📌 Conclusions

1. **Verbalized method** is simple and works reasonably well on both datasets, but its reliability depends heavily on the model’s instruction‑following ability (Qwen2.5-1.5B is acceptable; 0.5B fails).
2. **Self‑consistency** requires 10× more inference time. It performs best at moderate temperature (t=0.5) and on datasets with clear class boundaries (CLINC). On hard tasks (GRNTI), its advantage over verbalized is marginal.
3. **Latent information** methods (entropy, max prob) are the most efficient and effective, especially on well‑separated classes. They are also the fastest (no repeated sampling).
4. For practical use, we recommend **latent information** if you have white‑box access, otherwise **verbalized** with a sufficiently large model.

---

## 🚀 How to Reproduce

### Setup

```bash
git clone https://github.com/lastya24/ru-sci-llm-uncertainty-in-classification.git
cd ru-sci-llm-uncertainty-in-classification
pip install -r requirements.txt
