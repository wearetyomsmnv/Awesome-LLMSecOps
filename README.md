<p align="center">
  <img src="https://media.giphy.com/media/WPkNqX6qclrlG1LgxV/giphy.gif" alt="GIPHY Animation">
</p>


<div align="center">

# 🚀 Awesome LLMSecOps 

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
![GitHub stars](https://img.shields.io/github/stars/wearetyomsmnv/awesome-llmsecops?style=flat-square&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/wearetyomsmnv/awesome-llmsecops?style=flat-square&color=blue)
![GitHub last commit](https://img.shields.io/github/last-commit/wearetyomsmnv/awesome-llmsecops?style=flat-square&color=green)

🔐 A curated list of awesome resources for LLMSecOps (Large Language Model Security Operations) 🧠

### by @wearetyomsmnv


**Architecture | Vulnerabilities | Tools | Defense | Threat Modeling | Jailbreaks | RAG Security | PoC's | Study Resources | Books | Blogs | Datasets for Testing | OPS Security | Frameworks | Best Practices | Research | Tutorials | Companies | Community Resources**

</div>


>LLM safety is a huge body of knowledge that is important and relevant to society today. The purpose of this Awesome list is to provide the community with the necessary knowledge on how to build an LLM development process - safe, as well >as what threats may be encountered along the way. Everyone is welcome to contribute. 

> [!IMPORTANT]
>This repository, unlike many existing repositories, emphasizes the practical implementation of security and does not provide a lot of references to arxiv in the description.

---

<div align="center">

## 3 types of LLM architecture

![Group3](https://github.com/user-attachments/assets/0079c134-be60-42b0-afaa-a5df9bb7ece3)


<div align="center">

## Architecture risks


| Risk | Description |
|------|-------------|
| Recursive Pollution | LLMs can produce incorrect output with high confidence. If such output is used in training data, it can cause future LLMs to be trained on polluted data, creating a feedback loop problem. |
| Data Debt | LLMs rely on massive datasets, often too large to thoroughly vet. This lack of transparency and control over data quality presents a significant risk. |
| Black Box Opacity | Many critical components of LLMs are hidden in a "black box" controlled by foundation model providers, making it difficult for users to manage and mitigate risks effectively. |
| Prompt Manipulation | Manipulating the input prompts can lead to unstable and unpredictable LLM behavior. This risk is similar to adversarial inputs in other ML systems. |
| Poison in the Data | Training data can be contaminated intentionally or unintentionally, leading to compromised model integrity. This is especially problematic given the size and scope of data used in LLMs. |
| Reproducibility Economics | The high cost of training LLMs limits reproducibility and independent verification, leading to a reliance on commercial entities and potentially unreviewed models. |
| Model Trustworthiness | The inherent stochastic nature of LLMs and their lack of true understanding can make their output unreliable. This raises questions about whether they should be trusted in critical applications. |
| Encoding Integrity | Data is often processed and re-represented in ways that can introduce bias and other issues. This is particularly challenging with LLMs due to their unsupervised learning nature. |

</div>

**From [Berryville Institute of Machine Learning (BIML)](https://berryvilleiml.com/docs/BIML-LLM24.pdf) paper**


<div align="center">

## Vulnerabilities desctiption 
#### by Giskard

| Vulnerability | Description |
|---------------|-------------|
| Hallucination and Misinformation | These vulnerabilities often manifest themselves in the generation of fabricated content or the spread of false information, which can have far-reaching consequences such as disseminating misleading content or malicious narratives. |
| Harmful Content Generation | This vulnerability involves the creation of harmful or malicious content, including violence, hate speech, or misinformation with malicious intent, posing a threat to individuals or communities. |
| Prompt Injection | Users manipulating input prompts to bypass content filters or override model instructions can lead to the generation of inappropriate or biased content, circumventing intended safeguards. |
| Robustness | The lack of robustness in model outputs makes them sensitive to small perturbations, resulting in inconsistent or unpredictable responses that may cause confusion or undesired behavior. |
| Output Formatting | When model outputs do not align with specified format requirements, responses can be poorly structured or misformatted, failing to comply with the desired output format. |
| Information Disclosure | This vulnerability occurs when the model inadvertently reveals sensitive or private data about individuals, organizations, or entities, posing significant privacy risks and ethical concerns. |
| Stereotypes and Discrimination | If model's outputs are perpetuating biases, stereotypes, or discriminatory content, it leads to harmful societal consequences, undermining efforts to promote fairness, diversity, and inclusion. |


## LLMSecOps Life Cycle


![Group 2](https://github.com/user-attachments/assets/43a56dad-ddad-4097-a57e-aa035247810d)

</div>
<div align="center">

<h2>🛠 Tools for scanning</h2>

<table>
<tr>
<th>Tool</th>
<th>Description</th>
<th>Stars</th>
</tr>
<tr>
<td><a href="https://github.com/leondz/garak">🔧 Garak</a></td>
<td>LLM vulnerability scanner</td>
<td><img src="https://img.shields.io/github/stars/leondz/garak?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/prompt-security/ps-fuzz">🔧 ps-fuzz 2</a></td>
<td>Make your GenAI Apps Safe & Secure 🚀 Test & harden your system prompt</td>
<td><img src="https://img.shields.io/github/stars/prompt-security/ps-fuzz?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/pasquini-dario/LLMmap">🗺️ LLMmap</a></td>
<td>Tool for mapping LLM vulnerabilities</td>
<td><img src="https://img.shields.io/github/stars/pasquini-dario/LLMmap?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/msoedov/agentic_security">🛡️ Agentic Security</a></td>
<td>Security toolkit for AI agents</td>
<td><img src="https://img.shields.io/github/stars/msoedov/agentic_security?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/Mindgard/cli">🧠 Mindgard CLI</a></td>
<td>Command-line interface for Mindgard security tools</td>
<td><img src="https://img.shields.io/github/stars/Mindgard/cli?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/LostOxygen/llm-confidentiality">🔒 LLM Confidentiality</a></td>
<td>Tool for ensuring confidentiality in LLMs</td>
<td><img src="https://img.shields.io/github/stars/LostOxygen/llm-confidentiality?style=social" alt="GitHub stars"></td>
</tr>
   <tr>
<td><a href="https://github.com/Azure/PyRIT">🔒 PyRIT</a></td>
<td>The Python Risk Identification Tool for generative AI (PyRIT) is an open access automation framework to empower security professionals and machine learning engineers to proactively find risks in their generative AI systems.</td>
<td><img src="https://img.shields.io/github/stars/Azure/PyRIT?style=social" alt="GitHub stars"></td>
</tr>
</table>


> ### How to run garak
> 
> ```
> python -m pip install -U garak
> ```
> 
> Probe ChatGPT for encoding-based prompt injection (OSX/\*nix) (replace example value with a real OpenAI API key)
> 
> Probes is a simple .py file with prompts for LLM
> 
> **[Examples](https://github.com/leondz/garak/tree/main/garak/probes)**
>  
> ```
> export OPENAI_API_KEY="sk-123XXXXXXXXXXXX"
> python3 -m garak --model_type openai --model_name gpt-3.5-turbo --probes encoding
> ```
> 
> See if the Hugging Face version of GPT2 is vulnerable to DAN 11.0
> 
> ```
> python3 -m garak --model_type huggingface --model_name gpt2 --probes dan.Dan_11_0
> ```
> 
> **More examples on [Garak Tool](https://github.com/leondz/garak/blob/main/README.md#getting-started) instruction**



<h2>🛡️Defense</h2>

<table>
<tr>
<th>Tool</th>
<th>Description</th>
<th>Stars</th>
</tr>
<tr>
<td><a href="https://github.com/meta-llama/PurpleLlama">🛡️ PurpleLlama</a></td>
<td>Set of tools to assess and improve LLM security.</td>
<td><img src="https://img.shields.io/github/stars/meta-llama/PurpleLlama?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/protectai/rebuff">🛡️ Rebuff</a></td>
<td>API with built-in rules for identifying prompt injection and detecting data leakage through canary words.</td>
<td><img src="https://img.shields.io/github/stars/protectai/rebuff?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/laiyer-ai/llm-guard">🔒 LLM Guard</a></td>
<td>Self-hostable tool with multiple prompt and output scanners for various security issues.</td>
<td><img src="https://img.shields.io/github/stars/laiyer-ai/llm-guard?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/NVIDIA/NeMo-Guardrails">🚧 NeMo Guardrails</a></td>
<td>Tool that protects against jailbreak and hallucinations with customizable rulesets.</td>
<td><img src="https://img.shields.io/github/stars/NVIDIA/NeMo-Guardrails?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/deadbits/vigil-llm">👁️ Vigil</a></td>
<td>Offers dockerized and local setup options, using proprietary HuggingFace datasets for security detection.</td>
<td><img src="https://img.shields.io/github/stars/deadbits/vigil-llm?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/whylabs/langkit">🧰 LangKit</a></td>
<td>Provides functions for jailbreak detection, prompt injection, and sensitive information detection.</td>
<td><img src="https://img.shields.io/github/stars/whylabs/langkit?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/ShreyaR/guardrails">🛠️ GuardRails AI</a></td>
<td>Focuses on functionality, detects presence of secrets in responses.</td>
<td><img src="https://img.shields.io/github/stars/ShreyaR/guardrails?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://huggingface.co/Epivolis/Hyperion">🦸 Hyperion Alpha</a></td>
<td>Detects prompt injections and jailbreaks.</td>
<td>N/A</td>
</tr>
<tr>
<td><a href="https://github.com/protectai/llm-guard">🛡️ LLM-Guard</a></td>
<td>Tool for securing LLM interactions.</td>
<td><img src="https://img.shields.io/github/stars/protectai/llm-guard?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/Repello-AI/whistleblower">🚨 Whistleblower</a></td>
<td>Tool for detecting and preventing LLM vulnerabilities.</td>
<td><img src="https://img.shields.io/github/stars/Repello-AI/whistleblower?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/safellama/plexiglass">🔍 Plexiglass</a></td>
<td>Security tool for LLM applications.</td>
<td><img src="https://img.shields.io/github/stars/safellama/plexiglass?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/tldrsec/prompt-injection-defenses">🔍 Prompt Injection defenses</a></td>
<td>Rules for protected LLM</td>
<td><img src="https://img.shields.io/github/stars/tldrsec/prompt-injection-defenses?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://ai.raftds.ru/security/#">🔍 LLM Data Protector</a></td>
<td>Tools for protected LLM in chatbots</td>
</tr>
</table>

</div>


---
<div align="center">



## Threat Modeling

| Tool | Description |
|------|-------------|
| [Secure LLM Deployment: Navigating and Mitigating Safety Risks](https://arxiv.org/pdf/2406.11007) | Research paper on LLM security [sorry, but is really cool] |
| [ThreatModels](https://github.com/jsotiro/ThreatModels/tree/main) | Repository for LLM threat models |
| [Threat Modeling LLMs](https://aivillage.org/large%20language%20models/threat-modeling-llm/) | AI Village resource on threat modeling for LLMs |

![image](https://github.com/user-attachments/assets/0adcabdf-1afb-4ab2-aa8c-eef75c229842)
![image](https://github.com/user-attachments/assets/ed4340ad-ee95-47b3-8661-2660a2b0472e)


## Monitoring 

| Tool | Description |
|------|-------------|
|[Langfuse](https://langfuse.com/) | Open Source LLM Engineering Platform with security capabilities. |

## Watermarking

| Tool | Description |
|------|-------------|
| [MarkLLM](https://github.com/THU-BPM/MarkLLM) | An Open-Source Toolkit for LLM Watermarking. |

## Jailbreaks

| Resource | Description | Stars |
|----------|-------------|-------|
| [JailbreakBench](https://jailbreakbench.github.io/) | Website dedicated to evaluating and analyzing jailbreak methods for language models |
| [L1B3RT45](https://github.com/elder-plinius/L1B3RT45/) | GitHub repository containing information and tools related to AI jailbreaking |
| [llm-hacking-database](https://github.com/pdparchitect/llm-hacking-database)|This repository contains various attack against Large Language Models|
| [HaizeLabs jailbreak Database](https://launch.haizelabs.com/)| This database contains jailbreaks for multimodal language models|
| [Lakera PINT Benchmark](https://github.com/lakeraai/pint-benchmark) | A benchmark for prompt injection detection systems. | 
| [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) | An easy-to-use Python framework to generate adversarial jailbreak prompts | ![GitHub stars](https://img.shields.io/github/stars/EasyJailbreak/EasyJailbreak?style=social) |

## LLM Intrpretability

| Resource | Description |
|----------|-------------|
| [Интерпретируемость LLM](https://kolodezev.ru/interpretable_llm.html)| Dmitry Kolodezev's web page, which provides useful resources with LLM interpretation techniques| 

## PINT Benchmark scores (by lakera)

| Name | PINT Score | Test Date |
| ---- | ---------- | --------- |
| [Lakera Guard](https://lakera.ai/) | 98.0964% | 2024-06-12 |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | 91.5706% | 2024-06-12 |
| [Azure AI Prompt Shield for Documents](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection#prompt-shields-for-documents) | 91.1914% | 2024-04-05 |
| [Meta Prompt Guard](https://github.com/meta-llama/PurpleLlama/tree/main/Prompt-Guard) | 90.4496% | 2024-07-26 |
| [protectai/deberta-v3-base-prompt-injection](https://huggingface.co/protectai/deberta-v3-base-prompt-injection) | 88.6597% | 2024-06-12 |
| [WhyLabs LangKit](https://github.com/whylabs/langkit) | 80.0164% | 2024-06-12 |
| [Azure AI Prompt Shield for User Prompts](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection#prompt-shields-for-user-prompts) | 77.504% | 2024-04-05 |
| [Epivolis/Hyperion](https://huggingface.co/epivolis/hyperion) | 62.6572% | 2024-06-12 |
| [fmops/distilbert-prompt-injection](https://huggingface.co/fmops/distilbert-prompt-injection) | 58.3508% | 2024-06-12 |
| [deepset/deberta-v3-base-injection](https://huggingface.co/deepset/deberta-v3-base-injection) | 57.7255% | 2024-06-12 |
| [Myadav/setfit-prompt-injection-MiniLM-L3-v2](https://huggingface.co/myadav/setfit-prompt-injection-MiniLM-L3-v2) | 56.3973% | 2024-06-12 |


# Hallucinations Leaderboard

|Model|Hallucination Rate|Factual Consistency Rate|Answer Rate|Average Summary Length (Words)|
|----|----:|----:|----:|----:|
|GPT-4o|1.5 %|98.5 %|100.0 %|77.8|
|Zhipu AI GLM-4-9B-Chat|1.6 %|98.4 %|100.0 %|58.1|
|GPT-4o-mini|1.7 %|98.3 %|100.0 %|76.3|
|GPT-4-Turbo|1.7 %|98.3 %|100.0 %|86.2|
|GPT-4|1.8 %|98.2 %|100.0 %|81.1|
|GPT-3.5-Turbo|1.9 %|98.1 %|99.6 %|84.1|
|Microsoft Orca-2-13b|2.5 %|97.5 %|100.0 %|66.2|
|Intel Neural-Chat-7B-v3-3|2.7 %|97.3 %|100.0 %|60.7|
|Snowflake-Arctic-Instruct|3.0 %|97.0 %|100.0 %|68.7|
|Microsoft Phi-3-mini-128k-instruct|3.1 %|96.9 %|100.0 %|60.1|
|01-AI Yi-1.5-34B-Chat|3.9 %|96.1 %|100.0 %|83.7|
|Llama-3.1-405B-Instruct|3.9 %|96.1 %|99.6 %|85.7|
|Microsoft Phi-3-mini-4k-instruct|4.0 %|96.0 %|100.0 %|86.8|
|Llama-3-70B-Chat-hf|4.1 %|95.9 %|99.2 %|68.5|
|Mistral-Large2|4.4 %|95.6 %|100.0 %|77.4|
|Mixtral-8x22B-Instruct-v0.1|4.7 %|95.3 %|99.9 %|92.0|
|Qwen2-72B-Instruct|4.9 %|95.1 %|100.0 %|100.1|
|Llama-3.1-70B-Instruct|5.0 %|95.0 %|100.0 %|79.6|
|01-AI Yi-1.5-9B-Chat|5.0 %|95.0 %|100.0 %|85.7|
|Llama-3.1-8B-Instruct|5.5 %|94.5 %|100.0 %|71.0|
|Llama-2-70B-Chat-hf|5.9 %|94.1 %|99.9 %|84.9|
|Google Gemini-1.5-flash|6.6 %|93.4 %|98.1 %|62.8|
|Microsoft phi-2|6.7 %|93.3 %|91.5 %|80.8|
|Google Gemma-2-2B-it|7.0 %|93.0 %|100.0 %|62.2|
|Llama-3-8B-Chat-hf|7.4 %|92.6 %|99.8 %|79.7|
|Google Gemini-Pro|7.7 %|92.3 %|98.4 %|89.5|
|CohereForAI c4ai-command-r-plus|7.8 %|92.2 %|100.0 %|71.2|
|01-AI Yi-1.5-6B-Chat|8.2 %|91.8 %|100.0 %|98.9|
|databricks dbrx-instruct|8.3 %|91.7 %|100.0 %|85.9|
|Anthropic Claude-3-5-sonnet|8.6 %|91.4 %|100.0 %|103.0|
|Mistral-7B-Instruct-v0.3|9.8 %|90.2 %|100.0 %|98.4|
|Anthropic Claude-3-opus|10.1 %|89.9 %|95.5 %|92.1|
|Google Gemma-2-9B-it|10.1 %|89.9 %|100.0 %|70.2|
|Llama-2-13B-Chat-hf|10.5 %|89.5 %|99.8 %|82.1|
|Llama-2-7B-Chat-hf|11.3 %|88.7 %|99.6 %|119.9|
|Microsoft WizardLM-2-8x22B|11.7 %|88.3 %|99.9 %|140.8|
|Amazon Titan-Express|13.5 %|86.5 %|99.5 %|98.4|
|Google PaLM-2|14.1 %|85.9 %|99.8 %|86.6|
|Google Gemma-7B-it|14.8 %|85.2 %|100.0 %|113.0|
|Cohere-Chat|15.4 %|84.6 %|98.0 %|74.4|
|Anthropic Claude-3-sonnet|16.3 %|83.7 %|100.0 %|108.5|
|Google Gemma-1.1-7B-it|17.0 %|83.0 %|100.0 %|64.3|
|Anthropic Claude-2|17.4 %|82.6 %|99.3 %|87.5|
|Google Flan-T5-large|18.3 %|81.7 %|99.3|20.9|
|Cohere|18.9 %|81.1 %|99.8 %|59.8|
|Mixtral-8x7B-Instruct-v0.1|20.1 %|79.9 %|99.9 %|90.7|
|Apple OpenELM-3B-Instruct|24.8 %|75.2 %|99.3 %|47.2|
|Google Gemma-1.1-2B-it|27.8 %|72.2 %|100.0 %|66.8|
|Google Gemini-1.5-Pro|28.1 %|71.9 %|89.3 %|82.1|
|TII falcon-7B-instruct|29.9 %|70.1 %|90.0 %|75.5|

**From [this](https://github.com/vectara/hallucination-leaderboard) repo (update 5 aug)**



![image](https://github.com/user-attachments/assets/c051388f-9876-449b-81af-20308dfee4ac)

**This is a Safety Benchmark from [stanford university](https://crfm.stanford.edu/helm/air-bench/latest/)**
</div>

---

## RAG Security

| Resource | Description |
|----------|-------------|
| [Security Risks in RAG](https://ironcorelabs.com/security-risks-rag/) | Article on security risks in Retrieval-Augmented Generation (RAG) |
| [How RAG Poisoning Made LLaMA3 Racist](https://medium.com/m/global-identity-2?redirectUrl=https%3A%2F%2Fblog.repello.ai%2Fhow-rag-poisoning-made-llama3-racist-1c5e390dd564) | Blog post about RAG poisoning and its effects on LLaMA3 |
| [Adversarial AI - RAG Attacks and Mitigations](https://github.com/wearetyomsmnv/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/tree/main/ch15/RAG) | GitHub repository on RAG attacks, mitigations, and defense strategies |
| [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) | GitHub repository about poisoned RAG systems |
| [ConfusedPilot: Compromising Enterprise Information Integrity and Confidentiality with Copilot for Microsoft 365](https://arxiv.org/html/2408.04870v1) | Article about RAG vulnerabilities |
| [Awesome Jailbreak on LLMs - RAG Attacks](https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs?tab=readme-ov-file#attack-on-rag-based-llm) | Collection of RAG-based LLM attack techniques |

![image](https://github.com/user-attachments/assets/e0df02b1-9d7d-40ac-ba1b-b6f69ae68073)


## Agentic security 
| Tool | Description | Stars |
|------|-------------|-------|
| [invariant](https://github.com/invariantlabs-ai/invariant) | A trace analysis tool for AI agents. | ![GitHub stars](https://img.shields.io/github/stars/invariantlabs-ai/invariant?style=social) |
| [AgentBench](https://github.com/THUDM/AgentBench) | A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24) | ![GitHub stars](https://img.shields.io/github/stars/THUDM/AgentBench?style=social) |
| [Agent Hijacking, the thrue impact of prompt injection](https://dev.to/snyk/agent-hijacking-the-true-impact-of-prompt-injection-attacks-983) | Guide for attack langchain agents) | Article |
| [Damn Vulnerable Agent](https://github.com/WithSecureLabs/damn-vulnerable-llm-agent ) |Vulnerable LLM Agent | ![GitHub stars](https://img.shields.io/github/stars/WithSecureLabs/damn-vulnerable-llm-agent?style=social)  |
| [Agent Security Bench (ASB)](https://github.com/agiresearch/ASB)| Benchmark for agent security| ![GitHub stars](https://img.shields.io/github/stars/agiresearch/ASB?style=social)  |
| [Breaking Agents: Compromising Autonomous LLM Agents Through Malfunction Amplification](https://arxiv.org/pdf/2407.20859v1) | Research about typical agent vulnerabilities | Article |

## PoC

| Tool | Description | Stars |
|------|-------------|-------|
| [Visual Adversarial Examples](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models) | Jailbreaking Large Language Models with Visual Adversarial Examples | ![GitHub stars](https://img.shields.io/github/stars/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models?style=social) |
| [Weak-to-Strong Generalization](https://github.com/XuandongZhao/weak-to-strong) | Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision | ![GitHub stars](https://img.shields.io/github/stars/XuandongZhao/weak-to-strong?style=social) |
| [Image Hijacks](https://github.com/euanong/image-hijacks) | Repository for image-based hijacks of large language models | ![GitHub stars](https://img.shields.io/github/stars/euanong/image-hijacks?style=social) |
| [CipherChat](https://github.com/RobustNLP/CipherChat) | Secure communication tool for large language models | ![GitHub stars](https://img.shields.io/github/stars/RobustNLP/CipherChat?style=social) |
| [LLMs Finetuning Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) | Safety measures for fine-tuning large language models | ![GitHub stars](https://img.shields.io/github/stars/LLM-Tuning-Safety/LLMs-Finetuning-Safety?style=social) |
| [Virtual Prompt Injection](https://github.com/wegodev2/virtual-prompt-injection) | Tool for virtual prompt injection in language models | ![GitHub stars](https://img.shields.io/github/stars/wegodev2/virtual-prompt-injection?style=social) |
| [FigStep](https://github.com/ThuCCSLab/FigStep) | Jailbreaking Large Vision-language Models via Typographic Visual Prompts | ![GitHub stars](https://img.shields.io/github/stars/ThuCCSLab/FigStep?style=social) |
| [stealing-part-lm-supplementary](https://github.com/dpaleka/stealing-part-lm-supplementary) | Some code for "Stealing Part of a Production Language Model" | ![GitHub stars](https://img.shields.io/github/stars/dpaleka/stealing-part-lm-supplementary?style=social) |
| [Hallucination-Attack](https://github.com/PKU-YuanGroup/Hallucination-Attack) | Attack to induce LLMs within hallucinations | ![GitHub stars](https://img.shields.io/github/stars/PKU-YuanGroup/Hallucination-Attack?style=social) |
| [llm-hallucination-survey](https://github.com/HillZhang1999/llm-hallucination-survey) | Reading list of hallucination in LLMs. Check out our new survey paper: "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models" | ![GitHub stars](https://img.shields.io/github/stars/HillZhang1999/llm-hallucination-survey?style=social) |
| [LMSanitator](https://github.com/meng-wenlong/LMSanitator) | LMSanitator: Defending Large Language Models Against Stealthy Prompt Injection Attacks | ![GitHub stars](https://img.shields.io/github/stars/meng-wenlong/LMSanitator?style=social) |
| [Imperio](https://github.com/HKU-TASR/Imperio) | Imperio: Robust Prompt Engineering for Anchoring Large Language Models | ![GitHub stars](https://img.shields.io/github/stars/HKU-TASR/Imperio?style=social) |
| [Backdoor Attacks on Fine-tuned LLaMA](https://github.com/naimul011/backdoor_attacks_on_fine-tuned_llama) | Backdoor Attacks on Fine-tuned LLaMA Models | ![GitHub stars](https://img.shields.io/github/stars/naimul011/backdoor_attacks_on_fine-tuned_llama?style=social) |
| [CBA](https://github.com/MiracleHH/CBA) | Consciousness-Based Authentication for LLM Security | ![GitHub stars](https://img.shields.io/github/stars/MiracleHH/CBA?style=social) |
| [MuScleLoRA](https://github.com/ZrW00/MuScleLoRA) | A Framework for Multi-scenario Backdoor Fine-tuning of LLMs | ![GitHub stars](https://img.shields.io/github/stars/ZrW00/MuScleLoRA?style=social) |
| [BadActs](https://github.com/clearloveclearlove/BadActs) | BadActs: Backdoor Attacks against Large Language Models via Activation Steering | ![GitHub stars](https://img.shields.io/github/stars/clearloveclearlove/BadActs?style=social) |
| [TrojText](https://github.com/UCF-ML-Research/TrojText) | Trojan Attacks on Text Classifiers | ![GitHub stars](https://img.shields.io/github/stars/UCF-ML-Research/TrojText?style=social) |
| [AnyDoor](https://github.com/sail-sg/AnyDoor) | Create Arbitrary Backdoor Instances in Language Models | ![GitHub stars](https://img.shields.io/github/stars/sail-sg/AnyDoor?style=social) |
| [PromptWare](https://github.com/StavC/PromptWares) | A Jailbroken GenAI Model Can Cause Real Harm: GenAI-powered Applications are Vulnerable to PromptWares | ![GitHub stars](https://img.shields.io/github/stars/StavC/PromptWares?style=social) |
| [BrokenHill](https://github.com/BishopFox/BrokenHill) | Automated attack tool that generates crafted prompts to bypass restrictions in LLMs using greedy coordinate gradient (GCG) attack | ![GitHub stars](https://img.shields.io/github/stars/BishopFox/BrokenHill?style=social) |
| [LLaMator](https://github.com/RomiconEZ/LLaMator) | Framework for testing vulnerabilities of large language models with support for Russian language | ![GitHub stars](https://img.shields.io/github/stars/RomiconEZ/LLaMator?style=social) |
| [OWASP Agentic AI](https://github.com/precize/OWASP-Agentic-AI/) | OWASP Top 10 for Agentic AI (AI Agent Security) - Pre-release version | ![GitHub stars](https://img.shields.io/github/stars/precize/OWASP-Agentic-AI?style=social) |


---

## Study resource

| Tool | Description | 
|------|-------------|
| [Gandalf](https://gandalf.lakera.ai/) | Interactive LLM security challenge game |
| [Prompt Airlines](https://promptairlines.com/) | Platform for learning and practicing prompt engineering |
| [PortSwigger LLM Attacks](https://portswigger.net/web-security/llm-attacks/) | Educational resource on WEB LLM security vulnerabilities and attacks |
| [Invariant Labs CTF 2024](https://invariantlabs.ai/play-ctf-challenge-24) | CTF. Your should hack llm agentic |
| [DeepLearning.AI Red Teaming Course](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) | Short course on red teaming LLM applications |
| [Learn Prompting: Offensive Measures](https://learnprompting.org/docs/prompt_hacking/offensive_measures/) | Guide on offensive prompt engineering techniques |
| [Application Security LLM Testing](https://application.security/free/llm) | Free LLM security testing  |
| [Salt Security Blog: ChatGPT Extensions Vulnerabilities](https://salt.security/blog/security-flaws-within-chatgpt-extensions-allowed-access-to-accounts-on-third-party-websites-and-sensitive-data) | Article on security flaws in ChatGPT browser extensions |
| [safeguarding-llms](https://github.com/sshkhr/safeguarding-llms) | TMLS 2024 Workshop: A Practitioner's Guide To Safeguarding Your LLM Applications |
| [Damn Vulnerable LLM Agent](https://github.com/WithSecureLabs/damn-vulnerable-llm-agent) | Intentionally vulnerable LLM agent for security testing and education |
| [GPT Agents Arena](https://gpa.43z.one/) | Platform for testing and evaluating LLM agents in various scenarios |
| [AI Battle](https://play.secdim.com/game/ai-battle) | Interactive game focusing on AI security challenges |


![image](https://github.com/user-attachments/assets/17d3149c-acc2-48c9-a318-bda0b4c175ce)

## 📊 Community research articles

| Title | Authors | Year | 
|-------|---------|------|
| [📄 Bypassing Meta's LLaMA Classifier: A Simple Jailbreak](https://www.robustintelligence.com/blog-posts/bypassing-metas-llama-classifier-a-simple-jailbreak) | Robust Intelligence | 2024 |
| [📄 Vulnerabilities in LangChain Gen AI](https://unit42.paloaltonetworks.com/langchain-vulnerabilities/) | Unit42 | 2024 |
| [📄 Detecting Prompt Injection: BERT-based Classifier](https://labs.withsecure.com/publications/detecting-prompt-injection-bert-based-classifier) | WithSecure Labs | 2024 |
| [📄 Practical LLM Security: Takeaways From a Year in the Trenches](http://i.blackhat.com/BH-US-24/Presentations/US24-Harang-Practical-LLM-Security-Takeaways-From-Wednesday.pdf?_gl=1*1rlcqet*_gcl_au*MjA4NjQ5NzM4LjE3MjA2MjA5MTI.*_ga*OTQ0NTQ2MTI5LjE3MjA2MjA5MTM.*_ga_K4JK67TFYV*MTcyMzQwNTIwMS44LjEuMTcyMzQwNTI2My4wLjAuMA..&_ga=2.168394339.31932933.1723405201-944546129.1720620913) | NVIDIA | 2024 |
| [📄 Security ProbLLMs in xAI's Grok](https://embracethered.com/blog/posts/2024/security-probllms-in-xai-grok/) | Embrace The Red | 2024 |
| [📄 Persistent Pre-Training Poisoning of LLMs](https://spylab.ai/blog/poisoning-pretraining/) | SpyLab AI | 2024 |
| [📄 Navigating the Risks: A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents](https://arxiv.org/pdf/2411.09523) | Multiple Authors | 2024 |


## 🎓 Tutorials


| Resource | Description |
|----------|-------------|
| [📚 HADESS - Web LLM Attacks](https://hadess.io/web-llm-attacks/) | Understanding how to carry out web attacks using LLM |
| [📚 Red Teaming with LLMs](https://redteamrecipe.com/red-teaming-with-llms) | Practical methods for attacking AI systems |
| [📚 Lakera LLM Security](https://www.lakera.ai/blog/llm-security) | Overview of attacks on LLM |


<div align="center">

## 📚 Books

| 📖 Title | 🖋️ Author(s) | 🔍 Description |
|----------|--------------|----------------|
| [The Developer's Playbook for Large Language Model Security](https://www.amazon.com/Developers-Playbook-Large-Language-Security/dp/109816220X) | Steve Wilson  | 🛡️ Comprehensive guide for developers on securing LLMs |
| [Generative AI Security: Theories and Practices (Future of Business and Finance)](https://www.amazon.com/Generative-AI-Security-Theories-Practices/dp/3031542517) | Ken Huang, Yang Wang, Ben Goertzel, Yale Li, Sean Wright, Jyoti Ponnapalli | 🔬 In-depth exploration of security theories, laws, terms and practices in Generative AI |
|[Adversarial AI Attacks, Mitigations, and Defense Strategies: A cybersecurity professional's guide to AI attacks, threat modeling, and securing AI with MLSecOps](https://www.packtpub.com/en-ru/product/adversarial-ai-attacks-mitigations-and-defense-strategies-9781835087985)|John Sotiropoulos| Practical examples of code for your best mlsecops pipeline|




## BLOGS

| Blog |
|------|
| https://embracethered.com/blog/ |
| 🐦 https://twitter.com/llm_sec |
| 🐦 https://twitter.com/LLM_Top10 |
| 🐦 https://twitter.com/aivillage_dc |
| 🐦 https://twitter.com/elder_plinius/ |
| https://hiddenlayer.com/ |
| https://t.me/llmsecurity |


## DATA

| Resource | Description |
|----------|-------------|
| [Safety and privacy with Large Language Models](https://github.com/annjawn/llm-safety-privacy) | GitHub repository on LLM safety and privacy |
| [Jailbreak LLMs](https://github.com/verazuo/jailbreak_llms/tree/main/data) | Data for jailbreaking Large Language Models |
| [ChatGPT System Prompt](https://github.com/LouisShark/chatgpt_system_prompt) | Repository containing ChatGPT system prompts |
| [Do Not Answer](https://github.com/Libr-AI/do-not-answer) | Project related to LLM response control |
| [ToxiGen](https://github.com/microsoft/ToxiGen) | Microsoft dataset |
| [SafetyPrompts](https://safetyprompts.com/)| A Living Catalogue of Open Datasets for LLM Safety|
| [llm-security-prompt-injection](https://github.com/sinanw/llm-security-prompt-injection) | This project investigates the security of large language models by performing binary classification of a set of input prompts to discover malicious prompts. Several approaches have been analyzed using classical ML algorithms, a trained LLM model, and a fine-tuned LLM model. |

</div>

<div align="center">

## OPS 

![Group 4](https://github.com/user-attachments/assets/90133c33-ee58-4ec8-a9cb-c14fe529eb2f)


| Resource | Description |
|----------|-------------|
| https://sysdig.com/blog/llmjacking-stolen-cloud-credentials-used-in-new-ai-attack/ | LLMJacking: Stolen Cloud Credentials Used in New AI Attack |
| https://huggingface.co/docs/hub/security | Hugging Face Hub Security Documentation |
| https://developer.nvidia.com/blog/secure-llm-tokenizers-to-maintain-application-integrity/ | Secure LLM Tokenizers to Maintain Application Integrity |
| https://sightline.protectai.com/ | Sightline by ProtectAI <br><br>Check vulnerabilities on:<br>• Nemo by Nvidia<br>• Deep Lake<br>• Fine-Tuner AI<br>• Snorkel AI<br>• Zen ML<br>• Lamini AI<br>• Comet<br>• Titan ML<br>• Deepset AI<br>• Valohai<br><br>**For finding LLMops tools vulnerabilities** |
</div>

---

<div align="center">

## 🏗 Frameworks

<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://owasp.org/www-project-top-10-for-large-language-model-applications/"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>OWASP LLM TOP 10</b></sub></a><br />10 vulnerabilities for llm</td>
    <td align="center"><a href="https://owasp.org/www-project-top-10-for-large-language-model-applications/llm-top-10-governance-doc/LLM_AI_Security_and_Governance_Checklist-v1.pdf"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>LLM AI Cybersecurity & Governance Checklist 2</b></sub></a><br />Brief explanation</td>
    <td align="center"><a href="https://docs.google.com/document/d/1_F-1xp78LjyIiAwuO_II6enWBbOqKkYWFw2CpfZJ45U/edit?_bhlid=b838ad7e2c992ac7bb0133cb539a82a64b0c6ea5"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>LLMSecOps Cybersecurity Solution Landscape</b></sub></a><br />Brief explanation</td>
  </tr>
</table>
</div>


**LLMSECOPS, by OWASP**

![Group 12](https://github.com/user-attachments/assets/bf97f232-8532-450e-86bc-0ec39c5efe41)



## 💡 Best Practices

<table align="center"> <tr> <td align="center"> <h3>OWASP LLMSVS</h3> <p><strong>Large Language Model Security Verification Standard</strong></p> <p><a href="https://owasp.org/www-project-llm-verification-standard/">Project Link</a></p> </td> </tr> <tr> <td align="center"> <p>The primary aim of the OWASP LLMSVS Project is to provide an open security standard for systems which leverage artificial intelligence and Large Language Models.</p> <p>The standard provides a basis for designing, building, and testing robust LLM backed applications, including:</p> <ul style="list-style-type: none; padding: 0;"> <li>Architectural concerns</li> <li>Model lifecycle</li> <li>Model training</li> <li>Model operation and integration</li> <li>Model storage and monitoring</li> </ul> </td> </tr> </table> </div>


![image](https://github.com/user-attachments/assets/f5453935-f86a-401c-884c-14410d4c1a1c)

---

## 🌐 Community

<div align="center">

| Platform | Details |
|:--------:|---------|
| [OWASP SLACK](https://owasp.org/slack/invite) | **Channels:**<br>• #project-top10-for-llm<br>• #ml-risk-top5<br>• #project-ai-community<br>• #project-mlsec-top10<br>• #team-llm_ai-secgov<br>• #team-llm-redteam<br>• #team-llm-v2-brainstorm |
| [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security) | GitHub repository |
| [PWNAI](https://t.me/pwnai) | Telegram channel |
| [AiSec_X_Feed](https://t.me/aisecnews) | Telegram channel |
| [LVE_Project](https://lve-project.org/) | Official website |
| [Lakera AI Security resource hub](https://docs.google.com/spreadsheets/d/1tv3d2M4-RO8xJYiXp5uVvrvGWffM-40La18G_uFZlRM/edit?gid=639798153#gid=639798153) | Google Sheets document |
| [llm-testing-findings](https://github.com/BishopFox/llm-testing-findings/)| Templates with recomendation, cwe and other | 



| Name | LLM Security Company | URL |
|------|---------------------------|-----|
| Giskard | AI quality management system for ML models, focusing on vulnerabilities such as performance bias, hallucinations, and prompt injections. | https://www.giskard.ai/ |
| Lakera | Lakera Guard enhances LLM application security and counters a wide range of AI cyber threats. | https://www.lakera.ai/ |
| Lasso Security | Focuses on LLMs, offering security assessment, advanced threat modeling, and specialized training programs. | https://www.lasso.security/ |
| LLM Guard | Designed to strengthen LLM security, offers sanitization, malicious language detection, data leak prevention, and prompt injection resilience. | https://llmguard.com |
| LLM Fuzzer | Open-source fuzzing framework specifically designed for LLMs, focusing on integration into applications via LLM APIs. | https://github.com/llmfuzzer |
| Prompt Security | Provides a security, data privacy, and safety approach across all aspects of generative AI, independent of specific LLMs. | https://www.prompt.security |
| Rebuff | Self-hardening prompt injection detector for AI applications, using a multi-layered protection mechanism. | https://github.com/rebuff |
| Robust Intelligence | Provides AI firewall and continuous testing and evaluation. Creators of the airisk.io database donated to MITRE. | https://www.robustintelligence.com/ |
| WhyLabs | Protects LLMs from security threats, focusing on data leak prevention, prompt injection monitoring, and misinformation prevention. | https://www.whylabs.ai/ |
| [LLMbotomy: Shutting the Trojan Backdoors](http://i.blackhat.com/EU-24/Presentations/EU-24-Voros-LLMBotomyShuttingTheTrojanBackdoors.pdf) | BlackHat EU 2024: Novel approach to mitigate LLM Trojans through targeted noising of neurons |
| [Mind the Data Gap: Privacy Challenges in Autonomous AI Agents](http://i.blackhat.com/EU-24/Presentations/EU-24-Pappu-Mind-the-Data-Gap.pdf) | BlackHat EU 2024: Exploring key vulnerabilities in multi-agent AI systems |

</div>

## Benchmarks

| Resource | Description | Stars |
|----------|-------------|-------|
| [LLM Security Guidance Benchmarks](https://github.com/davisconsultingservices/llm_security_guidance_benchmarks) | Benchmarking lightweight, open-source LLMs for security guidance effectiveness using SECURE dataset | ![GitHub stars](https://img.shields.io/github/stars/davisconsultingservices/llm_security_guidance_benchmarks?style=social) |
| [SECURE](https://github.com/aiforsec/SECURE) | Benchmark for evaluating LLMs in cybersecurity scenarios, focusing on Industrial Control Systems | ![GitHub stars](https://img.shields.io/github/stars/aiforsec/SECURE?style=social) |
| [NIST AI TEVV](https://www.nist.gov/ai-test-evaluation-validation-and-verification-tevv) | AI Test, Evaluation, Validation and Verification framework by NIST | N/A |
| [Taming the Beast: Inside the Llama 3 Red Teaming Process](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20presentations/DEF%20CON%2032%20-%20Aaron%20Grattafiori%20Ivan%20Evtimov%20Joanna%20Bitton%20Maya%20Pavlova%20-%20Taming%20the%20Beast%20-%20Inside%20the%20Llama%203%20Red%20Team%20Process.pdf) | DEF CON 32 presentation on Llama 3 red teaming | 2024 |
