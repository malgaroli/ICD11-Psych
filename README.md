
# Multilingual Clinical Benchmark for Psychiatric Diagnosis

This repository accompanies the paper
**"A Multilingual Clinical Benchmark for Psychiatric Diagnosis: Large Language Models Achieve Clinician-Level Performance on WHO ICD-11 Field Studies."**

The repository provides code and prompts for reproducing results on a benchmark that evaluates large language models (LLMs) on psychiatric diagnostic reasoning. The benchmark uses expert-validated, multilingual clinical vignettes derived from three WHO ICD-11 field studies. The benchmark data will be released separately through a managed-access framework.

---

## Purpose of This Repository

Although large language models have demonstrated promising reasoning capabilities, their evaluation as diagnostic support tools has lacked robust, multilingual benchmarks validated against human expert performance. No such benchmark previously existed for multilingual psychiatric diagnosis. This repository addresses this gap by:

- Establishing a **multilingual psychiatric diagnostic benchmark** grounded in three WHO ICD-11 field studies
- Enabling **direct comparison between LLMs and human clinicians** under identical diagnostic instructions
- Providing **non-inferiority and equivalence testing** of LLM performance relative to clinicians
- Supporting **transparent, reproducible evaluation** of both open-weights and proprietary LLMs across languages, architectures, and model scales

The benchmark is designed to diagnostic performance across anxiety-, mood-, and stress-related disorders.

---

## Benchmark Overview

- **Clinical data source:** Three WHO ICD-11 case-vignette field studies (anxiety and fear-related disorders, mood disorders, disorders specifically associated with stress)
- **Clinicians:** 2,114 participants from the Global Clinical Practice Network (GCPN), spanning 155 countries across all WHO global regions
- **Clinical vignettes:** 43 expert-validated cases (11 anxiety-related, 21 mood-related, 11 stress-related)
- **Languages:** Chinese, English, French, Japanese, Russian, Spanish
- **Diagnostic categories:**
  - Anxiety and fear-related disorders 
  - Mood-related disorders 
  - Disorders specifically associated with stress
- **Models evaluated:** 10 LLMs — 7 open-weights (Gemma-3-27B-it, DeepSeek-R1-70B-Distill-Llama-70B, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, Mistral-7B-Instruct-v0.2, Mistral-Large-Instruct-2411, Qwen2.5-32B-Instruct) deployed in a HIPAA-compliant environment, and 3 proprietary models (Gemini-Pro-2.5, GPT-5.1, Claude-Opus-4.6) accessed through HIPAA-compliant API endpoints.

LLMs are evaluated using **zero-shot prompting** that mirrors the original ICD-11 clinician instructions. Models are additionally required to output ranked (Top-3) differential diagnoses to confirm that responses reflect clinical reasoning rather than stochastic guessing. **Deterministic sampling** is used for all open-weights models to ensure reproducible, auditable outputs.

---

## Prompting Strategy

- Prompts replicate the original WHO ICD-11 field study instructions provided to clinicians
- Included components:
  - Introductory text and diagnostic framing
  - Full clinical vignette text
  - Complete list of ICD-11 diagnostic categories for the relevant disorder grouping
  - Extended ICD-11 diagnostic guidelines were **omitted** from prompts due to token-length constraints; ablation analyses confirmed no substantial performance change
- For multilingual evaluation, instructions, vignettes, and model responses are all in the same target language.

All prompts used in the experiments are provided in the `prompts.md` file.

---

## Intended Use

 This repository is intended to support the empirical evaluation and future release of a benchmark for LLMs on multilingual psychiatric diagnosis, including:

- Measuring diagnostic accuracy and non-inferiority relative to clinicians
- Quantifying human–LLM diagnostic agreement
- Studying cross-lingual and cross-category diagnostic consistency


This tool is intended to support research on clinical decision-making and is not intended to replace professional medical advice. Always consult a qualified healthcare provider for any medical concerns.


---

## Funding

This work was supported by the National Institute of Mental Health (NIMH) through grants K23MH134068, R01MH129856, and R01MH117172; the National Heart, Lung, and Blood Institute (NHLBI) through grant R01HL156134; and the Betty and Gordon Moore Foundation. The content is solely the responsibility of the authors and does not represent the official views of the NIH, Moore Foundation, or the World Health Organization.

---

## Citation

If you use this benchmark, data, or codebase, please cite the associated paper.

---

## Contact


For questions, please contact:
- Matteo Malgaroli: matteo.malgaroli@nyulangone.org
