# AGAS vs. PAD-LLM Repository

This repository implements the adversarial benchmark between **AGAS (Agentic Group Attack System, ICDM 2026)** as a **Black-Box Shilling Attacker** and **PAD-LLM** as a **Black-Box Deep Neural Detector** with SHAP value interpretability.

---

## Architectural Workflow

### 1. Attacker: AGAS (Black-Box)
- **Zero Internal Access**: Manipulates target item recommendation ranks using black-box environment feedback.
- **Agentic Workers**: Deploys Profilers, Snipers, Camouflageurs, and Inactive workers to bypass rating variance filters.

### 2. Defender: PAD-LLM (Black-Box Deep Neural Model)
- **T5 Label Semantic Vector Extraction**:
  $$f_i = \\frac{\\sum a_{il} H_{i,l}}{\\sum a_{il} + \\epsilon}$$
- **3D Interaction Tensor Construction**: Combines user-item interaction matrix $R_{mix}'$ with label matrix $F'$ into a 3D tensor $T_1 = R'_{mix} \\odot F'$.
- **Dual-Branch Local-Global Feature Extraction**:
  - *Branch 1 (3D-DSConv)*: Captures local user-item-label neighborhood interactions.
  - *Branch 2 (Performer Attention)*: Captures global collaborative graph dependencies via kernelized self-attention ($\\phi(Q)(\\phi(K)^\\top V)$).
- **Gated Residual Fusion**:
  $$O_{fused} = (1 - \\sigma(\\gamma)) O_{CNN} + \\sigma(\\gamma) O_{Attn-CNN}$$
- **Joint Optimization**: Combines Focal Loss (class imbalance) with Contrastive Loss (hard negative mining).
- **SHAP Interpretability**: Generates SHAP value weight attributions to explain predictions despite black-box deep learning operations.

---

## Quickstart Guide

### 1. Repository Setup
```bash
git clone <repository-url>
cd agas-vs-padllm
pip install -r requirements.txt
pip install -e .
```

### 2. Run Benchmark Experiment
```bash
python experiments/run_simulation.py
```

### 3. Run Test Suite
```bash
python -m unittest discover tests/
```
