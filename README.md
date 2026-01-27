<p align="center">
  <img src="https://i.pinimg.com/736x/73/13/ff/7313ff4171a12334076a70b3c0854f4b.jpg" alt="LLMSecOps">
</p>

<div align="center">

# 🚀 Awesome LLMSecOps 

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
![GitHub stars](https://img.shields.io/github/stars/wearetyomsmnv/awesome-llmsecops?style=flat-square&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/wearetyomsmnv/awesome-llmsecops?style=flat-square&color=blue)
![GitHub last commit](https://img.shields.io/github/last-commit/wearetyomsmnv/awesome-llmsecops?style=flat-square&color=green)

🔐 A curated list of awesome resources for LLMSecOps (Large Language Model Security Operations) 🧠

### by @wearetyomsmnv and people

**Architecture | Vulnerabilities | Tools | Defense | Threat Modeling | Jailbreaks | RAG Security | PoC's | Study Resources | Books | Blogs | Datasets for Testing | OPS Security | Frameworks | Best Practices | Research | Tutorials | Companies | Community Resources**

</div>

>LLM safety is a huge body of knowledge that is important and relevant to society today. The purpose of this Awesome list is to provide the community with the necessary knowledge on how to build an LLM development process - safe, as well as what threats may be encountered along the way. Everyone is welcome to contribute. 

> [!IMPORTANT]
>This repository, unlike many existing repositories, emphasizes the practical implementation of security and does not provide a lot of references to arxiv in the description.

<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 2em 0;">

<div align="center">

## Architecture risks

*Overview of fundamental architectural risks and challenges in LLM systems.*

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

## Vulnerabilities description 
#### by Giskard

*Common vulnerabilities and security issues found in LLM applications.*

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

*Security scanning and vulnerability assessment tools for LLM applications.*

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
<td><a href="https://github.com/LostOxygen/llm-confidentiality">🔒 LLM Confidentiality</a></td>
<td>Tool for ensuring confidentiality in LLMs</td>
<td><img src="https://img.shields.io/github/stars/LostOxygen/llm-confidentiality?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/Azure/PyRIT">🔒 PyRIT</a></td>
<td>The Python Risk Identification Tool for generative AI (PyRIT) is an open access automation framework to empower security professionals and machine learning engineers to proactively find risks in their generative AI systems.</td>
<td><img src="https://img.shields.io/github/stars/Azure/PyRIT?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/promptfoo/promptfoo">🔧 promptfoo</a></td>
<td>LLM red teaming and evaluation framework. Test for jailbreaks, prompt injection, and other vulnerabilities with adversarial attacks (PAIR, tree-of-attacks, crescendo). CI/CD integration.</td>
<td><img src="https://img.shields.io/github/stars/promptfoo/promptfoo?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/RomiconEZ/LLaMator">🔧 LLaMator</a></td>
<td>Framework for testing vulnerabilities of large language models with support for Russian language</td>
<td><img src="https://img.shields.io/github/stars/RomiconEZ/LLaMator?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/ReversecLabs/spikee">🔧 Spikee</a></td>
<td>Comprehensive testing framework for LLM applications. Tests prompt injection, jailbreaks, and other vulnerabilities. Supports custom targets, attacks, judges, and guardrail evaluation</td>
<td><img src="https://img.shields.io/github/stars/ReversecLabs/spikee?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/KOKOSde/localmod">🛡️ LocalMod</a></td>
<td>Self-hosted content moderation API with prompt injection detection, toxicity filtering, PII detection, and NSFW filtering. Runs 100% offline.</td>
<td><img src="https://img.shields.io/github/stars/KOKOSde/localmod?style=social" alt="GitHub stars"></td>
</tr>
</table>

<h2>🛡️Defense</h2>

*Defensive mechanisms, guardrails, and security controls for protecting LLM applications.*

<div align="center">

## Security by Design

| Category | Method / Technology | Principle of Operation (Mechanism) | Examples of Use / Developers |
|----------|---------------------|-----------------------------------|------------------------------|
| **1. Fundamental Alignment** | **RLHF (Reinforcement Learning from Human Feedback)** | Training a model with reinforcement learning based on a reward model, which is trained on human evaluations. It optimizes for "usefulness" and "safety." | OpenAI (GPT-4), Yandex (YandexGPT) |
| | **DPO (Direct Preference Optimization)** | Direct optimization of response probabilities based on preference pairs, bypassing the creation of a separate reward model. It is described as more stable and effective. | Meta (Llama 3), Mistral, open models |
| | **Constitutional AI / RLAIF** | Using the model itself to criticize and correct its responses according to a set of rules ("Constitution"). AI replaces human labeling (RLAIF). | Anthropic (Claude 3) |
| **2. Internal Control (Interpretability)** | **Representation Engineering (RepE)** | Detection and suppression of neuron activation vectors responsible for undesirable concepts (e.g., falsehood, lust for power) in real-time. | Center for AI Safety (CAIS) |
| | **Circuit Breakers** | Redirection ("short-circuiting") of internal representations of malicious queries into orthogonal space, causing failure or nonsense. | GraySwan AI, researchers |
| | **Machine Unlearning** | Algorithmic "erasure" of dangerous knowledge or protected data from model weights (e.g., via Gradient Ascent) so that the model physically "forgets" them. | Research groups, Microsoft |
| **3. External Filters (Guardrails)** | **Llama Guard** | A specialized LLM-classifier that checks incoming prompts and outgoing responses for compliance with a risk taxonomy (MLCommons). | Meta |
| | **NeMo Guardrails** | A programmable dialogue management system. It uses the Colang language for strict topic adherence and attack blocking. | NVIDIA |
| | **Prompt Guard / Shields** | Lightweight models (based on BERT/DeBERTA) for detecting jailbreaks and prompt injections before they reach the LLM. | Meta, Azure AI |
| | **SmoothLLM** | A randomized smoothing method: creating copies of a prompt with symbolic perturbations to disrupt the structure of adversarial attacks (e.g., GCG suffixes). | Researchers (SmoothLLM authors) |
| | **Google Safety Filters** | Multi-level content filtering with customizable sensitivity thresholds and semantic vector analysis. | Google (Gemini API) |
| **4. System Instructions** | **System Prompts / Tags** | Using special tokens (e.g., `</start_header_id>`) to separate system and user instructions. | OpenAI, Meta, Anthropic |
| | **Instruction Hierarchy** | Prioritizing system instructions over user instructions to protect against prompt injection, especially when the model learns to ignore "forget past instructions" commands. | OpenAI (GPT-4o Mini) |
| **5. Testing (Red Teaming)** | **Automated Attacks (GCG, AutoDAN)** | Using algorithms and other LLMs to generate hundreds of thousands of adversarial prompts to find vulnerabilities. | Research groups |

</div>

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
<td>API with built-in rules for identifying prompt injection and detecting data leakage through canary words. <em>(ProtectAI is now part of Palo Alto Networks)</em></td>
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
<td>Tool for securing LLM interactions. <em>(ProtectAI is now part of Palo Alto Networks)</em></td>
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
<td>N/A</td>
</tr>
<tr>
<td><a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/responsible-ai/gemini_prompt_attacks_mitigation_examples.ipynb">🔍 Gen AI & LLM Security for developers: Prompt attack mitigations on Gemini</a></td>
<td>Security tool for LLM applications.</td>
<td><img src="https://img.shields.io/github/stars/GoogleCloudPlatform/generative-ai/?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://github.com/NeuralTrust/TrustGate">🔍 TrustGate</a></td>
<td>Generative Application Firewall that detects and blocks attacks against GenAI Applications.</td>
<td><img src="https://img.shields.io/github/stars/NeuralTrust/TrustGate?style=social" alt="GitHub stars"></td>
</tr>
<tr>
<tr>
<td><a href="https://github.com/tenuo-ai/tenuo">🛡️ Tenuo</a></td>
<td>Capability tokens for AI agents with task-scoped TTLs, offline verification and proof-of-possession binding.</td>
<td><img src="https://img.shields.io/github/stars/tenuo-ai/tenuo?style=social)" alt="GitHub stars"></td>
</tr>
<tr>
<td><a href="https://edward-playground.github.io/aidefense-framework/">🛡️ AIDEFEND</a></td>
<td>Practical knowledge base for AI security defenses. Based on MAESTRO framework, MITRE D3FEND, ATLAS, ATT&CK, Google Secure AI Framework, and OWASP Top 10 LLM 2025/ML Security 2023.</td>
<td>N/A</td>
</tr>
</table>

</div>

<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 2em 0;">

<div align="center">



## Threat Modeling

*Frameworks and methodologies for identifying and modeling threats in LLM systems.*

| Tool | Description |
|------|-------------|
| [Secure LLM Deployment: Navigating and Mitigating Safety Risks](https://arxiv.org/pdf/2406.11007) | Research paper on LLM security [sorry, but is really cool] |
| [ThreatModels](https://github.com/jsotiro/ThreatModels/tree/main) | Repository for LLM threat models |
| [Threat Modeling LLMs](https://aivillage.org/large%20language%20models/threat-modeling-llm/) | AI Village resource on threat modeling for LLMs |
| [Sberbank AI Cybersecurity Threat Model](https://www.sberbank.ru/ru/person/kibrary/experts/model-ugroz-kiberbezopasnosti-ai) | Sberbank's threat model for AI cybersecurity |
| [Pangea Attack Taxonomy](https://pangea.cloud/resources/taxonomy/) | Comprehensive taxonomy of AI/LLM attacks and vulnerabilities | Pangea |

![image](https://github.com/user-attachments/assets/0adcabdf-1afb-4ab2-aa8c-eef75c229842)
![image](https://github.com/user-attachments/assets/ed4340ad-ee95-47b3-8661-2660a2b0472e)

## Monitoring 

*Tools and platforms for monitoring LLM applications, detecting anomalies, and tracking security events.*

| Tool | Description |
|------|-------------|
|[Langfuse](https://langfuse.com/) | Open Source LLM Engineering Platform with security capabilities. |
|[HiveTrace](https://hivetrace.ru/preview/) | LLM monitoring and security platform for GenAI applications. Detects prompt injection, jailbreaks, malicious HTML/Markdown elements, and PII. Provides real-time anomaly detection and security alerts. |

## Watermarking

*Tools and techniques for watermarking LLM-generated content to detect AI-generated text.*

| Tool | Description |
|------|-------------|
| [MarkLLM](https://github.com/THU-BPM/MarkLLM) | An Open-Source Toolkit for LLM Watermarking. |

## Jailbreaks

*Resources, databases, and benchmarks for understanding and testing jailbreak techniques against LLMs.*

| Resource | Description | Stars |
|----------|-------------|-------|
| [JailbreakBench](https://jailbreakbench.github.io/) | Website dedicated to evaluating and analyzing jailbreak methods for language models | N/A |
| [L1B3RT45](https://github.com/elder-plinius/L1B3RT45/) | GitHub repository containing information and tools related to AI jailbreaking | ![GitHub stars](https://img.shields.io/github/stars/elder-plinius/L1B3RT45?style=social) |
| [llm-hacking-database](https://github.com/pdparchitect/llm-hacking-database)|This repository contains various attacks against Large Language Models| ![GitHub stars](https://img.shields.io/github/stars/pdparchitect/llm-hacking-database?style=social) |
| [HaizeLabs jailbreak Database](https://launch.haizelabs.com/)| This database contains jailbreaks for multimodal language models| N/A |
| [Lakera PINT Benchmark](https://github.com/lakeraai/pint-benchmark) | A comprehensive benchmark for prompt injection detection systems. Evaluates detection systems across multiple categories (prompt injection, jailbreak, hard negatives, chat, documents) and supports evaluation in 20+ languages. Open-source benchmark with Jupyter notebook for custom evaluations. | ![GitHub stars](https://img.shields.io/github/stars/lakeraai/pint-benchmark?style=social) |
| [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak) | An easy-to-use Python framework to generate adversarial jailbreak prompts | ![GitHub stars](https://img.shields.io/github/stars/EasyJailbreak/EasyJailbreak?style=social) |

## LLM Interpretability

*Resources for understanding and interpreting LLM behavior, decision-making, and internal mechanisms.*

| Resource | Description |
|----------|-------------|
| [Интерпретируемость LLM](https://kolodezev.ru/interpretable_llm.html)| Dmitry Kolodezev's web page, which provides useful resources with LLM interpretation techniques |

## PINT Benchmark scores (by lakera)

*Prompt Injection Test (PINT) benchmark scores comparing different prompt injection detection systems.*

| Name | PINT Score | Test Date |
| ---- | ---------- | --------- |
| [Lakera Guard](https://lakera.ai/) | 95.2200% | 2025-05-02 |
| [Azure AI Prompt Shield for Documents](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection#prompt-shields-for-documents) | 89.1241% | 2025-05-02 |
| [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | 79.1366% | 2025-05-02 |
| [Llama Prompt Guard 2 (86M)](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | 78.7578% | 2025-05-05 |
| [Google Model Armor](https://cloud.google.com/security-command-center/docs/model-armor-overview) | 70.0664% | 2025-08-27 |
| [Aporia Guardrails](https://www.aporia.com/) | 66.4373% | 2025-05-02 |
| [Llama Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M) | 61.8168% | 2025-05-02 |

> **Note:** ProtectAI is now part of Palo Alto Networks

# Hallucinations Leaderboard

![Top 25 Hallucination Rates](https://raw.githubusercontent.com/vectara/hallucination-leaderboard/main/img/top25_hallucination_rates_2025-12-09.png)

> **Note:** For the complete and most up-to-date interactive leaderboard, visit the [Hugging Face leaderboard](https://huggingface.co/spaces/vectara/hallucination-leaderboard) or the [GitHub repository](https://github.com/vectara/hallucination-leaderboard).

**From [this](https://github.com/vectara/hallucination-leaderboard) repo (last updated: December 18, 2025)**



![image](https://github.com/user-attachments/assets/c051388f-9876-449b-81af-20308dfee4ac)

**This is a Safety Benchmark from [Stanford University](https://crfm.stanford.edu/helm/air-bench/latest/)**
</div>

<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 2em 0;">

## RAG Security

*Security considerations, attacks, and defenses for Retrieval-Augmented Generation (RAG) systems.*

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

*Security tools, benchmarks, and research focused on autonomous AI agents and their vulnerabilities.*

| Tool | Description | Stars |
|------|-------------|-------|
| [invariant](https://github.com/invariantlabs-ai/invariant) | A trace analysis tool for AI agents. | ![GitHub stars](https://img.shields.io/github/stars/invariantlabs-ai/invariant?style=social) |
| [AgentBench](https://github.com/THUDM/AgentBench) | A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24) | ![GitHub stars](https://img.shields.io/github/stars/THUDM/AgentBench?style=social) |
| [Agent Hijacking, the true impact of prompt injection](https://dev.to/snyk/agent-hijacking-the-true-impact-of-prompt-injection-attacks-983) | Guide for attack langchain agents | Article |
| [Breaking Agents: Compromising Autonomous LLM Agents Through Malfunction Amplification](https://arxiv.org/pdf/2407.20859v1) | Research about typical agent vulnerabilities | Article |
| [Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers](https://arxiv.org/html/2506.13538v2) | First large-scale empirical study of MCP servers security and maintainability | Article |
| [Awesome MCP Security](https://github.com/Puliczek/awesome-mcp-security) | Curated list of MCP security resources | ![GitHub stars](https://img.shields.io/github/stars/Puliczek/awesome-mcp-security?style=social) |
| [Awesome LLM Agent Security](https://github.com/wearetyomsmnv/Awesome-LLM-agent-Security) | Comprehensive collection of LLM agent security resources, attacks, vulnerabilities | ![GitHub stars](https://img.shields.io/github/stars/wearetyomsmnv/Awesome-LLM-agent-Security?style=social) |
| [MCP Security Analysis](https://arxiv.org/pdf/2511.03841) | Research paper on MCP security vulnerabilities and analysis | Article |
| [Tenuo](https://github.com/tenuo-ai/tenuo) | Capability-based authorization framework for AI agents. Task-scoped warrants with cryptographic attenuation, PoP binding, offline verification. LangChain/LangGraph/MCP integrations. | ![GitHub stars](https://img.shields.io/github/stars/tenuo-ai/tenuo?style=social) |


## Agentic Browser Security

*Security research and analysis of AI-powered browser agents and their unique attack vectors.*

| Resource | Description | Source |
|----------|-------------|--------|
| [From Inbox to Wipeout: Perplexity Comet's AI Browser Quietly Erasing Google Drive](https://www.straiker.ai/blog/from-inbox-to-wipeout-perplexity-comets-ai-browser-quietly-erasing-google-drive) | Research on zero-click Google Drive wiper attack via Perplexity Comet. Shows how polite, well-structured emails can trigger destructive actions in agentic browsers. | Straiker STAR Labs |
| [Agentic Browser Security Analysis](https://arxiv.org/html/2506.07153v2) | Research paper on security vulnerabilities in agentic browsers | Article |
| [Browser AI Agents: The New Weakest Link](https://labs.sqrx.com/browser-ai-agents-the-new-weakest-link-22a38a552d7f) | Analysis of security risks in browser-based AI agents | Sqrx Labs |
| [Comet Prompt Injection Vulnerability](https://brave.com/blog/comet-prompt-injection/) | Brave's analysis of prompt injection vulnerabilities in Perplexity Comet browser | Brave |

## PoC

*Proof of Concept implementations demonstrating various LLM attacks, vulnerabilities, and security research.*

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
| [OWASP Agentic AI](https://github.com/precize/OWASP-Agentic-AI/) | OWASP Top 10 for Agentic AI (AI Agent Security) - Pre-release version | ![GitHub stars](https://img.shields.io/github/stars/precize/OWASP-Agentic-AI?style=social) |

<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 2em 0;">

## Study resource

*Educational platforms, CTF challenges, courses, and training resources for learning LLM security.*

| Tool | Description | 
|------|-------------|
| [Gandalf](https://gandalf.lakera.ai/) | Interactive LLM security challenge game |
| [Prompt Airlines](https://promptairlines.com/) | Platform for learning and practicing prompt engineering |
| [PortSwigger LLM Attacks](https://portswigger.net/web-security/llm-attacks/) | Educational resource on WEB LLM security vulnerabilities and attacks |
| [Invariant Labs CTF 2024](https://invariantlabs.ai/play-ctf-challenge-24) | CTF. You should hack LLM agentic |
| [Invariant Labs CTF Summer 24](https://huggingface.co/spaces/invariantlabs/ctf-summer-24/tree/main) | Hugging Face Space with CTF challenges |
| [Crucible](https://crucible.dreadnode.io/) | LLM security training platform |
| [Poll Vault CTF](http://poll-vault.chal.hackthe.vote/) | CTF challenge with ML/LLM components |
| [MyLLMDoc](https://myllmdoc.com/) | LLM security training platform |
| [AI CTF PHDFest2 2025](https://aictf.phdays.fun/) | AI CTF competition from PHDFest2 2025 |
| [AI in Security](https://aiinsec.ru/) | Russian platform for AI security training |
| [DeepLearning.AI Red Teaming Course](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) | Short course on red teaming LLM applications |
| [Learn Prompting: Offensive Measures](https://learnprompting.org/docs/prompt_hacking/offensive_measures/) | Guide on offensive prompt engineering techniques |
| [Application Security LLM Testing](https://application.security/free/llm) | Free LLM security testing  |
| [Salt Security Blog: ChatGPT Extensions Vulnerabilities](https://salt.security/blog/security-flaws-within-chatgpt-extensions-allowed-access-to-accounts-on-third-party-websites-and-sensitive-data) | Article on security flaws in ChatGPT browser extensions |
| [safeguarding-llms](https://github.com/sshkhr/safeguarding-llms) | TMLS 2024 Workshop: A Practitioner's Guide To Safeguarding Your LLM Applications |
| [Damn Vulnerable LLM Agent](https://github.com/WithSecureLabs/damn-vulnerable-llm-agent) | Intentionally vulnerable LLM agent for security testing and education |
| [GPT Agents Arena](https://gpa.43z.one/) | Platform for testing and evaluating LLM agents in various scenarios |
| [AI Battle](https://play.secdim.com/game/ai-battle) | Interactive game focusing on AI security challenges |
| [AI/LLM Exploitation Challenges](https://academy.8ksec.io/course/ai-exploitation-challenges) | Challenges to test your knowledge of AI, ML, and LLMs |
| [TryHackMe AI/ML Security Threats](https://medium.com/genai-llm-security/tryhackme-ai-ml-security-threats-walkthrough-writeup-04abd3f717ca) | Walkthrough and writeup for TryHackMe AI/ML Security Threats room | Article |

![image](https://github.com/user-attachments/assets/17d3149c-acc2-48c9-a318-bda0b4c175ce)

## 📊 Community research articles

*Research articles, security advisories, and technical papers from the security community.*

| Title | Authors | Year | 
|-------|---------|------|
| [📄 Bypassing Meta's LLaMA Classifier: A Simple Jailbreak](https://www.robustintelligence.com/blog-posts/bypassing-metas-llama-classifier-a-simple-jailbreak) | Robust Intelligence | 2024 |
| [📄 Vulnerabilities in LangChain Gen AI](https://unit42.paloaltonetworks.com/langchain-vulnerabilities/) | Unit42 | 2024 |
| [📄 Detecting Prompt Injection: BERT-based Classifier](https://labs.withsecure.com/publications/detecting-prompt-injection-bert-based-classifier) | WithSecure Labs | 2024 |
| [📄 Practical LLM Security: Takeaways From a Year in the Trenches](http://i.blackhat.com/BH-US-24/Presentations/US24-Harang-Practical-LLM-Security-Takeaways-From-Wednesday.pdf?_gl=1*1rlcqet*_gcl_au*MjA4NjQ5NzM4LjE3MjA2MjA5MTI.*_ga*OTQ0NTQ2MTI5LjE3MjA2MjA5MTM.*_ga_K4JK67TFYV*MTcyMzQwNTIwMS44LjEuMTcyMzQwNTI2My4wLjAuMA..&_ga=2.168394339.31932933.1723405201-944546129.1720620913) | NVIDIA | 2024 |
| [📄 Security ProbLLMs in xAI's Grok](https://embracethered.com/blog/posts/2024/security-probllms-in-xai-grok/) | Embrace The Red | 2024 |
| [📄 Persistent Pre-Training Poisoning of LLMs](https://spylab.ai/blog/poisoning-pretraining/) | SpyLab AI | 2024 |
| [📄 Navigating the Risks: A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents](https://arxiv.org/pdf/2411.09523) | Multiple Authors | 2024 |
| [📄 Practical AI Agent Security](https://ai.meta.com/blog/practical-ai-agent-security/) | Meta | 2025 |
| [📄 Security Advisory: Anthropic's Slack MCP Server Vulnerable to Data Exfiltration](https://embracethered.com/blog/posts/2025/security-advisory-anthropic-slack-mcp-server-data-leakage/) | Embrace The Red | 2025 |

## 🎓 Tutorials

*Step-by-step guides and tutorials for understanding and implementing LLM security practices.*

| Resource | Description |
|----------|-------------|
| [📚 HADESS - Web LLM Attacks](https://hadess.io/web-llm-attacks/) | Understanding how to carry out web attacks using LLM |
| [📚 Red Teaming with LLMs](https://redteamrecipe.com/red-teaming-with-llms) | Practical methods for attacking AI systems |
| [📚 Lakera LLM Security](https://www.lakera.ai/blog/llm-security) | Overview of attacks on LLM |

<div align="center">

## 📚 Books

*Comprehensive books covering LLM security, adversarial AI, and secure AI development practices.*

| 📖 Title | 🖋️ Author(s) | 🔍 Description |
|----------|--------------|----------------|
| [The Developer's Playbook for Large Language Model Security](https://www.amazon.com/Developers-Playbook-Large-Language-Security/dp/109816220X) | Steve Wilson  | 🛡️ Comprehensive guide for developers on securing LLMs |
| [Generative AI Security: Theories and Practices (Future of Business and Finance)](https://www.amazon.com/Generative-AI-Security-Theories-Practices/dp/3031542517) | Ken Huang, Yang Wang, Ben Goertzel, Yale Li, Sean Wright, Jyoti Ponnapalli | 🔬 In-depth exploration of security theories, laws, terms and practices in Generative AI |
|[Adversarial AI Attacks, Mitigations, and Defense Strategies: A cybersecurity professional's guide to AI attacks, threat modeling, and securing AI with MLSecOps](https://www.packtpub.com/en-ru/product/adversarial-ai-attacks-mitigations-and-defense-strategies-9781835087985)|John Sotiropoulos| Practical examples of code for your best mlsecops pipeline|




## BLOGS

*Security blogs, Twitter feeds, and Telegram channels focused on AI/LLM security.*

### Websites & Twitter

| Resource | Description |
|----------|-------------|
| [Embrace The Red](https://embracethered.com/blog/) | Blog on AI security, red teaming, and LLM vulnerabilities |
| [HiddenLayer](https://hiddenlayer.com/) | AI security company blog |
| [CyberArk](https://www.cyberark.com/blog) | Blog on AI agents, identity risks, and security |
| [Straiker](https://www.straiker.ai/blog) | AI security research and agentic browser security |
| [Firetail](https://www.firetail.ai/blog) | LLM security, prompt injection, and AI vulnerabilities |
| [Palo Alto Networks](https://www.paloaltonetworks.com/blog) | Unit 42 research on AI security and agentic AI attacks |
| [Trail of Bits](https://blog.trailofbits.com) | Security research including AI/ML pickle file security |
| [NCSC](https://www.ncsc.gov.uk/blog) | UK National Cyber Security Centre blog on AI safeguards |
| [Knostic](https://www.knostic.ai/blog) | AI Security Posture Management (AISPM) |
| [0din](https://0din.ai/blog) | Secure LLM and RAG deployment practices |
| [@llm_sec](https://twitter.com/llm_sec) | Twitter feed on LLM security |
| [@LLM_Top10](https://twitter.com/LLM_Top10) | Twitter feed on OWASP LLM Top 10 |
| [@aivillage_dc](https://twitter.com/aivillage_dc) | AI Village Twitter |
| [@elder_plinius](https://twitter.com/elder_plinius/) | Twitter feed on AI security |

### Telegram Channels

| Channel | Language | Description |
|---------|----------|-------------|
| [PWN AI](https://t.me/pwnai) | RU | Practical AI Security and MLSecOps: LLM security, agents, guardrails, real-world threats |
| [Борис_ь с ml](https://t.me/borismlsec) | RU | Machine Learning + Information Security: ML, data science and cyber/AI security |
| [Евгений Кокуйкин — Raft](https://t.me/kokuykin) | RU | Building Raft AI and GPT-based applications: trust & safety, reliability and security |
| [LLM Security](https://t.me/llmsecurity) | RU | Focused on LLM security: jailbreaks, prompt injection, adversarial attacks, benchmarks |
| [AISecHub](https://t.me/AISecHub) | EN | Global AI security hub: curated research, articles, reports and tools |
| [AI Security Lab](https://t.me/aisecuritylab) | RU | Laboratory by Raft x ITMO University: breaking and defending AI systems |
| [ML&Sec Feed](https://t.me/mlsecfeed) | RU/EN | Aggregated feed for ML & security: news, tools, research links |
| [AISec [x_feed]](https://t.me/aisecnews) | RU/EN | Digest of AI security content from X, blogs and papers |
| [AI SecOps](https://t.me/aisecops) | RU | AI Security Operations: monitoring, incident response, SIEM/SOC integrations |
| [OK ML](https://t.me/okmlai) | RU | ML/DS/AI channel with focus on repositories, tools and vulnerabilities |
| [AI Attacks](https://t.me/aiattacks) | EN | Stream of AI attack examples and threat intelligence |
| [AGI Security](https://t.me/agisec) | EN | Artificial General Intelligence Security discussions |

## DATA

*Datasets for testing LLM security, prompt injection examples, and safety evaluation data.*

| Resource | Description |
|----------|-------------|
| [Safety and privacy with Large Language Models](https://github.com/annjawn/llm-safety-privacy) | GitHub repository on LLM safety and privacy |
| [Jailbreak LLMs](https://github.com/verazuo/jailbreak_llms/tree/main/data) | Data for jailbreaking Large Language Models |
| [ChatGPT System Prompt](https://github.com/LouisShark/chatgpt_system_prompt) | Repository containing ChatGPT system prompts |
| [Do Not Answer](https://github.com/Libr-AI/do-not-answer) | Project related to LLM response control |
| [ToxiGen](https://github.com/microsoft/ToxiGen) | Microsoft dataset |
| [SafetyPrompts](https://safetyprompts.com/)| A Living Catalogue of Open Datasets for LLM Safety|
| [llm-security-prompt-injection](https://github.com/sinanw/llm-security-prompt-injection) | This project investigates the security of large language models by performing binary classification of a set of input prompts to discover malicious prompts. Several approaches have been analyzed using classical ML algorithms, a trained LLM model, and a fine-tuned LLM model. |
| [Prompt Injections Dataset](https://github.com/Dmtr-Dr/prompt-injections) | Dataset containing prompt injection examples for testing and research | ![GitHub stars](https://img.shields.io/github/stars/Dmtr-Dr/prompt-injections?style=social) |

</div>

<div align="center">

## OPS 

*Operational security considerations: supply chain risks, infrastructure vulnerabilities, and production deployment security.*

![Group 4](https://github.com/user-attachments/assets/90133c33-ee58-4ec8-a9cb-c14fe529eb2f)

| Resource | Description |
|----------|-------------|
| https://sysdig.com/blog/llmjacking-stolen-cloud-credentials-used-in-new-ai-attack/ | LLMJacking: Stolen Cloud Credentials Used in New AI Attack |
| https://huggingface.co/docs/hub/security | Hugging Face Hub Security Documentation |
| https://github.com/ShenaoW/awesome-llm-supply-chain-security | LLM Supply chain security resources|
| https://developer.nvidia.com/blog/secure-llm-tokenizers-to-maintain-application-integrity/ | Secure LLM Tokenizers to Maintain Application Integrity |
| https://sightline.protectai.com/ | Sightline by ProtectAI <em>(ProtectAI is now part of Palo Alto Networks)</em><br><br>Check vulnerabilities on:<br>• Nemo by Nvidia<br>• Deep Lake<br>• Fine-Tuner AI<br>• Snorkel AI<br>• Zen ML<br>• Lamini AI<br>• Comet<br>• Titan ML<br>• Deepset AI<br>• Valohai<br><br>**For finding LLMops tools vulnerabilities** |
| [ShadowMQ: How Code Reuse Spread Critical Vulnerabilities Across the AI Ecosystem](https://share.google/rBYXiMlAMeE4XCt2I) | Research on critical RCE vulnerabilities in AI inference servers (Meta Llama Stack, NVIDIA TensorRT-LLM, vLLM, SGLang, Modular) caused by unsafe ZeroMQ and pickle deserialization | Oligo Security |
</div>

<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 2em 0;">

<div align="center">

## 🏗 Frameworks

*Comprehensive security frameworks, standards, and governance models for LLM and AI security.*

<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://genai.owasp.org/llm-top-10/"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>OWASP Top 10 for LLM Applications 2025 (v2.0)</b></sub></a><br />Updated list including System Prompt Leakage, Vector and Embedding Weaknesses</td>
    <td align="center"><a href="https://genai.owasp.org/llm-top-10/"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>OWASP Top 10 for Agentic Applications (2026 Edition)</b></sub></a><br />First industry standard for autonomous AI agent risks (released Dec 2025)</td>
    <td align="center"><a href="https://owasp.org/www-project-ai-testing-guide/"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>OWASP AI Testing Guide v1</b></sub></a><br />Open standard for testing AI system trustworthiness (Nov 2025)</td>
  </tr>
  <tr>
    <td align="center"><a href="https://genai.owasp.org/resource/owasp-genai-security-project-solutions-reference-guide-q2_q325/"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>GenAI Security Solutions Reference Guide</b></sub></a><br />Vendor-neutral guide for GenAI security architecture (Q2-Q3 2025)</td>
    <td align="center"><a href="https://owasp.org/www-project-top-10-for-large-language-model-applications/llm-top-10-governance-doc/LLM_AI_Security_and_Governance_Checklist-v1.pdf"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>LLM AI Cybersecurity & Governance Checklist</b></sub></a><br />Security and governance checklist</td>
    <td align="center"><a href="https://docs.google.com/document/d/1_F-1xp78LjyIiAwuO_II6enWBbOqKkYWFw2CpfZJ45U/edit?_bhlid=b838ad7e2c992ac7bb0133cb539a82a64b0c6ea5"><img src="https://owasp.org/assets/images/logo.png" width="100px;" alt=""/><br /><sub><b>LLMSecOps Cybersecurity Solution Landscape</b></sub></a><br />Solution landscape overview</td>
  </tr>
</table>
</div>

> **All OWASP GenAI Resources:** [genai.owasp.org/resources/](https://genai.owasp.org/resources/)

**LLMSECOPS, by OWASP**

![Group 12](https://github.com/user-attachments/assets/bf97f232-8532-450e-86bc-0ec39c5efe41)

### Additional Security Frameworks

| Framework | Organization | Description |
|-----------|--------------|-------------|
| [MCP Security Governance](https://github.com/CloudSecurityAlliance/mcp-security-governance) | Cloud Security Alliance | Governance framework for the Model Context Protocol ecosystem. Developing policies, standards, and assessment tools for secure MCP server deployment. |
| [Databricks AI Security Framework (DASF) 2.0](https://www.databricks.com/resources/whitepaper/databricks-ai-security-framework-dasf) | Databricks | Actionable framework for managing AI security. Includes 62 security risks across three stages and 64 controls applicable to any data and AI platform. |
| [Google Secure AI Framework (SAIF) 2.0](https://saif.google/) | Google | Secure AI Framework focused on agents. Practitioner-focused framework for building powerful agents users can trust. |
| [Snowflake AI Security Framework](https://www.snowflake.com/en/resources/white-paper/snowflake-ai-security-framework/) | Snowflake | Comprehensive framework for securing AI deployments on Snowflake platform. |

## AI Security Solutions Radar

![2025 AI Security Solutions Radar](https://www.riskinsight-wavestone.com/wp-content/uploads/2025/09/Illustration-1.png)

> **Source:** [2025 AI Security Solutions Radar](https://www.riskinsight-wavestone.com/en/2025/09/2025-ai-security-solutions-radar/) by RiskInsight-Wavestone

</div>

<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 2em 0;">

## 🌐 Community

*Community resources, platforms, and collaborative spaces for LLM security practitioners.*

<div align="center">

| Platform | Details |
|:--------:|---------|
| [OWASP SLACK](https://owasp.org/slack/invite) | **Channels:**<br>• #project-top10-for-llm<br>• #ml-risk-top5<br>• #project-ai-community<br>• #project-mlsec-top10<br>• #team-llm_ai-secgov<br>• #team-llm-redteam<br>• #team-llm-v2-brainstorm |
| [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security) | GitHub repository |
| [Awesome AI Security Telegram](https://github.com/ivolake/awesome-ai-security-tg) | Curated list of Telegram channels and chats on AI Security, AI/MLSecOps, LLM Security | ![GitHub stars](https://img.shields.io/github/stars/ivolake/awesome-ai-security-tg?style=social) |
| [LVE_Project](https://lve-project.org/) | Official website |
| [Lakera AI Security resource hub](https://docs.google.com/spreadsheets/d/1tv3d2M4-RO8xJYiXp5uVvrvGWffM-40La18G_uFZlRM/edit?gid=639798153#gid=639798153) | Google Sheets document |
| [llm-testing-findings](https://github.com/BishopFox/llm-testing-findings/)| Templates with recommendations, CWE and other | 
| [Arcanum Prompt Injection Taxonomy](https://github.com/Arcanum-Sec/arc_pi_taxonomy/tree/main) | Structured taxonomy of prompt injection attacks categorizing attack intents, techniques, and evasions. Resource for security researchers, AI developers, and red teamers. | ![GitHub stars](https://img.shields.io/github/stars/Arcanum-Sec/arc_pi_taxonomy?style=social) |

</div>

## Benchmarks

*Security benchmarks, evaluation frameworks, and standardized tests for assessing LLM security capabilities.*

| Resource | Description | Stars |
|----------|-------------|-------|
| [Backbone Breaker Benchmark (b3)](https://www.lakera.ai/blog/the-backbone-breaker-benchmark) | Human-grounded benchmark for testing AI agent security. Built by Lakera with UK AI Security Institute using 194,000+ human attack attempts from Gandalf: Agent Breaker. Tests backbone LLM resilience across 10 threat snapshots. | Article |
| [Backbone Breaker Benchmark Paper](https://arxiv.org/html/2508.18106v3) | Research paper on the Backbone Breaker Benchmark methodology and findings | Article |
| [CyberSoCEval](https://ai.meta.com/research/publications/cybersoceval-benchmarking-llms-capabilities-for-malware-analysis-and-threat-intelligence-reasoning/) | Meta's benchmark for evaluating LLM capabilities in malware analysis and threat intelligence reasoning | Meta Research |
| [Agent Security Bench (ASB)](https://github.com/agiresearch/ASB) | Benchmark for agent security | ![GitHub stars](https://img.shields.io/github/stars/agiresearch/ASB?style=social) |
| [AI Safety Benchmark](https://sproutnan.github.io/AI-Safety_Benchmark/) | Comprehensive benchmark for AI safety evaluation | N/A |
| [AI Safety Benchmark Paper](https://arxiv.org/abs/2506.14697) | Research paper on AI safety benchmarking methodologies | Article |
| [Evaluating Prompt Injection Datasets](https://hiddenlayer.com/innovation-hub/evaluating-prompt-injection-datasets/) | Analysis and evaluation framework for prompt injection datasets | HiddenLayer |
| [LLM Security Guidance Benchmarks](https://github.com/davisconsultingservices/llm_security_guidance_benchmarks) | Benchmarking lightweight, open-source LLMs for security guidance effectiveness using SECURE dataset | ![GitHub stars](https://img.shields.io/github/stars/davisconsultingservices/llm_security_guidance_benchmarks?style=social) |
| [SECURE](https://github.com/aiforsec/SECURE) | Benchmark for evaluating LLMs in cybersecurity scenarios, focusing on Industrial Control Systems | ![GitHub stars](https://img.shields.io/github/stars/aiforsec/SECURE?style=social) |
| [NIST AI TEVV](https://www.nist.gov/ai-test-evaluation-validation-and-verification-tevv) | AI Test, Evaluation, Validation and Verification framework by NIST | N/A |
| [Taming the Beast: Inside the Llama 3 Red Teaming Process](https://media.defcon.org/DEF%20CON%2032/DEF%20CON%2032%20presentations/DEF%20CON%2032%20-%20Aaron%20Grattafiori%20Ivan%20Evtimov%20Joanna%20Bitton%20Maya%20Pavlova%20-%20Taming%20the%20Beast%20-%20Inside%20the%20Llama%203%20Red%20Team%20Process.pdf) | DEF CON 32 presentation on Llama 3 red teaming | 2024 |
