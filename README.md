# Multilingual Clinical Benchmark for Psychiatric Diagnosis

This repository accompanies the paper  
**“A Multilingual Clinical Benchmark for Psychiatric Diagnosis: Large Language Models Achieve Clinician-Level Performance on WHO ICD-11 Field Studies.”**

It provides a reproducible benchmark for evaluating large language models (LLMs) on psychiatric diagnostic reasoning using expert-validated, multilingual clinical vignettes derived from WHO ICD-11 field studies.

---

## Purpose of This Repository

Inadequate access to mental health assessment remains a critical global barrier, particularly in regions with limited availability of trained clinicians. Although large language models have demonstrated promising reasoning capabilities, their evaluation as diagnostic support tools has lacked robust, multilingual benchmarks validated against clinician performance.

This repository addresses this gap by:

- Establishing a **multilingual psychiatric diagnostic benchmark** grounded in WHO ICD-11 field studies  
- Enabling **direct comparison between LLMs and human clinicians** under identical diagnostic instructions  
- Supporting **transparent, reproducible evaluation** of open-weight LLMs across languages, architectures, and model scales  

The benchmark is designed to test whether LLMs can achieve *clinician-level diagnostic performance* across anxiety-, mood-, and stress-related disorders.

---

## Benchmark Overview

- **Clinical data source:** WHO ICD-11 field studies  
- **Clinicians:** 2,699 participants  
- **Clinical vignettes:** 43 expert-validated cases  
- **Languages:** Chinese, English, French, Japanese, Russian, Spanish  
- **Diagnostic categories:**  
  - Anxiety-related disorders  
  - Mood-related disorders  
  - Stress-related disorders  

LLMs are evaluated using zero-shot prompting that mirrors the original ICD-11 clinician instructions. Models are required to output ranked (Top-3) differential diagnoses to assess diagnostic reasoning beyond chance-level guessing.

---

## Prompting Strategy

- Prompts replicate the **original ICD-11 field study instructions** provided verbatim to clinicians  
- Included components:
  - Task introduction and diagnostic framing  
  - Full clinical vignette text  
  - Complete list of ICD-11 diagnostic categories  
- A structured `[ROLE]` definition (psychiatric diagnostician) and `[FORMAT]` block enforce standardized outputs  
- For multilingual evaluation, models are instructed to first translate the vignette into English and then perform the diagnostic task to standardize reasoning across languages  

All prompts used in the experiments are provided in the `prompts.md` file.

---

## Intended Use

This repository is intended for:

- Benchmarking new LLMs on multilingual psychiatric diagnosis  
- Studying cross-lingual diagnostic consistency  
- Measuring agreement between clinicians and AI systems  
- Supporting responsible development of clinical decision-support tools  

This benchmark is **not intended for direct clinical deployment**.

---

## Funding

This work was supported by the National Institute of Mental Health (NIMH), the National Heart, Lung, and Blood Institute (NHLBI), and the Betty and Gordon Moore Foundation.

---

## Citation

If you use this benchmark, data, or codebase, please cite the associated paper.


## Contact

**Disclaimer:** This tool is intended to support clinical decision-making and not to replace professional medical advice. Always consult a qualified healthcare provider for any medical concerns.

For questions or support, please contact matteo.malgaroli@nyulangone.org

