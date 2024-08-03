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




LLM safety is a huge body of knowledge that is important and relevant to society today. The purpose of this Awesome list is to provide the community with the necessary knowledge on how to build an LLM development process - safe, as well as what threats may be encountered along the way. Everyone is welcome to contribute. 

This repository, unlike many existing repositories, emphasizes the practical implementation of security and does not provide a lot of references to arxiv in the description.

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
|[Langfuse](https://langfuse.com/)|Open Source LLM Engineering Platform with security capabilities|

## Watermarking

https://github.com/THU-BPM/MarkLLM

## Jailbreaks

| Resource | Description |
|----------|-------------|
| [JailbreakBench](https://jailbreakbench.github.io/) | Website dedicated to evaluating and analyzing jailbreak methods for language models |
| [L1B3RT45](https://github.com/elder-plinius/L1B3RT45/) | GitHub repository containing information and tools related to AI jailbreaking |
| [llm-hacking-database](https://github.com/pdparchitect/llm-hacking-database)|This repository contains various attack against Large Language Models.|


# Hallucinations Leaderboard

| Model | Hallucination Rate | Factual Consistency Rate | Answer Rate | Average Summary Length (Words) |
|-------|:------------------:|:------------------------:|:-----------:|:-------------------------------:|
| GPT 4 Turbo | 2.5% | 97.5% | 100.0% | 86.2 |
| Snowflake Arctic | 2.6% | 97.4% | 100.0% | 68.7 |
| Intel Neural Chat 7B | 2.8% | 97.2% | 89.5% | 57.6 |
| 01-AI Yi-1.5-34B-Chat | 3.0% | 97.0% | 100.0% | 83.7 |
| GPT 4 | 3.0% | 97.0% | 100.0% | 81.1 |
| GPT 4o mini | 3.1% | 96.9% | 100.0% | 76.3 |
| Microsoft Orca-2-13b | 3.2% | 96.8% | 100.0% | 66.2 |
| Qwen2-72B-Instruct | 3.5% | 96.5% | 100.0% | 100.1 |
| GPT 3.5 Turbo | 3.5% | 96.5% | 99.6% | 84.1 |
| Mistral-Large2 | 3.6% | 96.4% | 100.0% | 77.4 |
| 01-AI Yi-1.5-9B-Chat | 3.7% | 96.3% | 100.0% | 85.0 |

**From [this](https://github.com/vectara/hallucination-leaderboard) repo (update 25 july)**

</div>

---

## RAG Security

| Resource | Description |
|----------|-------------|
| [Security Risks in RAG](https://ironcorelabs.com/security-risks-rag/) | Article on security risks in Retrieval-Augmented Generation (RAG) |
| [How RAG Poisoning Made LLaMA3 Racist](https://medium.com/m/global-identity-2?redirectUrl=https%3A%2F%2Fblog.repello.ai%2Fhow-rag-poisoning-made-llama3-racist-1c5e390dd564) | Blog post about RAG poisoning and its effects on LLaMA3 |
| [Adversarial AI - RAG Attacks and Mitigations](https://github.com/wearetyomsmnv/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/tree/main/ch15/RAG) | GitHub repository on RAG attacks, mitigations, and defense strategies |
| [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) | GitHub repository about poisoned RAG systems |

![image](https://github.com/user-attachments/assets/e0df02b1-9d7d-40ac-ba1b-b6f69ae68073)


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
| [stealing-part-lm-supplementary](https://github.com/dpaleka/stealing-part-lm-supplementary)|Some code for "Stealing Part of a Production Language Model"|![GitHub stars](https://img.shields.io/github/stars/dpaleka/stealing-part-lm-supplementary?style=social)|
|[Hallucination-Attack](https://github.com/PKU-YuanGroup/Hallucination-Attack)|Attack to induce LLMs within hallucinations|![GitHub stars](https://img.shields.io/github/stars/PKU-YuanGroup/Hallucination-Attack?style=social)|
|[llm-hallucination-survey](https://github.com/HillZhang1999/llm-hallucination-survey)|Reading list of hallucination in LLMs. Check out our new survey paper: "Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models"|![GitHub stars](https://img.shields.io/github/stars/HillZhang1999/llm-hallucination-survey?style=social)|

---

## Study resource

| Tool | Description | 
|------|-------------|
| [Gandalf](https://gandalf.lakera.ai/) | Interactive LLM security challenge game |
| [Prompt Airlines](https://promptairlines.com/) | Platform for learning and practicing prompt engineering |
| [PortSwigger LLM Attacks](https://portswigger.net/web-security/llm-attacks/) | Educational resource on WEB LLM security vulnerabilities and attacks |
| [DeepLearning.AI Red Teaming Course](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) | Short course on red teaming LLM applications |
| [Learn Prompting: Offensive Measures](https://learnprompting.org/docs/prompt_hacking/offensive_measures/) | Guide on offensive prompt engineering techniques |
| [Application Security LLM Testing](https://application.security/free/llm) | Free LLM security testing  |
| [Salt Security Blog: ChatGPT Extensions Vulnerabilities](https://salt.security/blog/security-flaws-within-chatgpt-extensions-allowed-access-to-accounts-on-third-party-websites-and-sensitive-data) | Article on security flaws in ChatGPT browser extensions |
|[safeguarding-llms](https://github.com/sshkhr/safeguarding-llms)|TMLS 2024 Workshop: A Practitioner's Guide To Safeguarding Your LLM Applications|
|[Damn Vulnerable LLM Agent]( https://github.com/WithSecureLabs/damn-vulnerable-llm-agent)|n/a|


![image](https://github.com/user-attachments/assets/17d3149c-acc2-48c9-a318-bda0b4c175ce)

## 📊 Community research articles

| Title | Authors | Year | 
|-------|---------|------|
| [📄 Bypassing Meta’s LLaMA Classifier: A Simple Jailbreak](https://www.robustintelligence.com/blog-posts/bypassing-metas-llama-classifier-a-simple-jailbreak) | Robust Intelligence | 2024 |
| [📄 Vulnerabilities in LangChain Gen AI](https://unit42.paloaltonetworks.com/langchain-vulnerabilities/) | Unit42 | 2024 |

## 🎓 Tutorials

1. [📚 HADESS - Web LLM Attacks](https://hadess.io/web-llm-attacks/)
   - Understand how u can do attack in web via llm
2. [📚 Red Teaming with LLMs](https://redteamrecipe.com/red-teaming-with-llms)
   - Practical Techniques for attacking ai systems
3. [📚 Lakera LLM Security](https://www.lakera.ai/blog/llm-security)
   - Overwiev for attacks on llm

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
| https://t.me/pwnai |
| https://github.com/sinanw/llm-security-prompt-injection |

## DATA

| Resource | Description |
|----------|-------------|
| [Safety and privacy with Large Language Models](https://github.com/annjawn/llm-safety-privacy) | GitHub repository on LLM safety and privacy |
| [Jailbreak LLMs](https://github.com/verazuo/jailbreak_llms/tree/main/data) | Data for jailbreaking Large Language Models |
| [ChatGPT System Prompt](https://github.com/LouisShark/chatgpt_system_prompt) | Repository containing ChatGPT system prompts |
| [Do Not Answer](https://github.com/Libr-AI/do-not-answer) | Project related to LLM response control |
| [ToxiGen](https://github.com/microsoft/ToxiGen) | Microsoft dataset |
| [SafetyPrompts](https://safetyprompts.com/)| A Living Catalogue of Open Datasets for LLM Safety|

</div>

<div align="center">

## OPS 

![image](https://github.com/user-attachments/assets/e7fe456e-4dc5-447c-90b4-392844d938e9)

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
  </tr>
</table>
</div>



## 💡 Best Practices

<table align="center"> <tr> <td align="center"> <h3>OWASP LLMSVS</h3> <p><strong>Large Language Model Security Verification Standard</strong></p> <p><a href="https://owasp.org/www-project-llm-verification-standard/">Project Link</a></p> </td> </tr> <tr> <td align="center"> <p>The primary aim of the OWASP LLMSVS Project is to provide an open security standard for systems which leverage artificial intelligence and Large Language Models.</p> <p>The standard provides a basis for designing, building, and testing robust LLM backed applications, including:</p> <ul style="list-style-type: none; padding: 0;"> <li>Architectural concerns</li> <li>Model lifecycle</li> <li>Model training</li> <li>Model operation and integration</li> <li>Model storage and monitoring</li> </ul> </td> </tr> </table> </div>

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



| Name | LLM Security Company | URL |
|------|---------------------------|-----|
| CalypsoAI Moderator | Focuses on preventing data leakage, full auditability, and malicious code detection. | https://www.prompt.security |
| Giskard | AI quality management system for ML models, focusing on vulnerabilities such as performance bias, hallucinations, and prompt injections. | https://www.giskard.ai/ |
| Lakera | Lakera Guard enhances LLM application security and counters a wide range of AI cyber threats. | https://www.lakera.ai/ |
| Lasso Security | Focuses on LLMs, offering security assessment, advanced threat modeling, and specialized training programs. | https://www.lasso.security/ |
| LLM Guard | Designed to strengthen LLM security, offers sanitization, malicious language detection, data leak prevention, and prompt injection resilience. | https://llmguard.com |
| LLM Fuzzer | Open-source fuzzing framework specifically designed for LLMs, focusing on integration into applications via LLM APIs. | https://github.com/llmfuzzer |
| Prompt Security | Provides a security, data privacy, and safety approach across all aspects of generative AI, independent of specific LLMs. | https://www.prompt.security |
| Rebuff | Self-hardening prompt injection detector for AI applications, using a multi-layered protection mechanism. | https://github.com/rebuff |
| Robust Intelligence | Provides AI firewall and continuous testing and evaluation. Creators of the airisk.io database donated to MITRE. | https://www.whylabs.ai/ |
| WhyLabs | Protects LLMs from security threats, focusing on data leak prevention, prompt injection monitoring, and misinformation prevention. | https://www.whylabs.ai/

</div>





