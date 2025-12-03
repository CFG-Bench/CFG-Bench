# CFG-Bench

<div align="center">

<h2>Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents</h2>

[![arXiv](https://img.shields.io/badge/cs.CV-2511.18685-b31b1b?logo=arxiv&logoColor=red)](https://www.arxiv.org/abs/2511.18685)

</div>

---

## News
* **`2025.11.24`** 🚀 **Paper Release**: We officially release CFG-Bench, a Fine-Grained Cognitive Benchmark for Embodied AI.
* **`Coming Soon`** ⏳ **Data Release**: The full benchmark package, including fine-grained action annotations, QA benchmark, and the curated video manifest, will be released soon.

## Introduction

Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents.

## Task Demonstration

<p align="center">
    <img src="./docs/demonstration.png" width="96%">
</p>

## Results

- **Model Comparison:**

<p align="center">
    <img src="./docs/model_comparision.png" width="96%">
</p>

- **Benchmark Comparison:**

<p align="center">
    <img src="./docs/benchmark_comparison.png" width="96%">
</p>


- **Benchmark Statistics:**

<p align="center">
    <img src="./docs/data_statistic_1.png" width="96%">
</p>
Data statistics of CFG-Bench. (a) Distribution and video length statistics of the five datasets. (b) The distribution of tasks across four tiers. AW means average words of questions.

## Citation

If you find our work helpful for your research, please consider citing our work.

```bibtex
@article{liu2025beyond,
  title={Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents},
  author={Liu, Dayong and Xu, Chao and Chen, Weihong and Zhang, Suyu and Wang, Juncheng and Deng, Jiankang and Sun, Baigui and Liu, Yang},
  journal={arXiv preprint arXiv:2511.18685},
  year={2025}
}
```
