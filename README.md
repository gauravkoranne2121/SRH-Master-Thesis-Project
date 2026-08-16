# SRH-Master-Thesis-Project
# Hybrid NLP & Conversational AI Framework for Automated Skill Extraction

> **Master's Thesis Research Project**  
> An automated framework combining Transformer-based Named Entity Recognition (JobBERT), Large Language Models (OpenAI GPT / Anthropic Claude), and interactive chat streaming to extract, categorize, and quantify market skills from job postings.

---

## 📌 Abstract / Overview

Extracting structured skill demands from unstructured job postings is critical for labor market analytics and automated recruitment matching. This thesis introduces a hybrid pipeline that combines:
1. **Domain-Specific Named Entity Recognition (NER)**: Uses `jjzha/jobbert_skill_extraction` alongside `spaCy` to identify entity spans in raw text.
2. **Rule-Based Post-Processing**: Filters generic noise, stop words, and low-confidence tokens.
3. **Generative LLM Refinement**: Refines, contextualizes, and categorizes entities into three standard categories: **Technical Skills**, **Non-Technical / Soft Skills**, and **Tools & Technologies**.
4. **Interactive Conversational AI Interface**: Allows recruiters and job seekers to query job descriptions, upload documents, and stream extracted skills in real time.

---

## 🏗 System Architecture

```text
 Unstructured Job Postings (CSV / Web Upload)
                │
                ▼
  [ spaCy Tokenization & Truncation ]
                │
                ▼
  [ JobBERT Token Classification (NER) ] ──► Initial Span Extraction
                │
                ▼
  [ Rule-Based Text Cleaning ] ──► Filtering Stop Words & Noise
                │
                ▼
  [ OpenAI / Claude LLM Layer ] ──► Context Refinement & Deduplication
                │
                ▼
  [ Interactive Chat / Batch Analytics ] ──► Streaming Chat UI or CSV Metrics
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Environment Variables

Set your API keys:

```bash
# Linux/macOS
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Windows (Command Prompt)
set OPENAI_API_KEY="your-openai-key"
set ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Execution

Run batch analysis over a CSV dataset:

```bash
python main.py
```

---

## 📊 Sample Batch Output

```text
Most Common Technical Skills:
- machine learning: 142
- data modeling: 118
- statistical analysis: 95

Most Common Non-Technical Skills:
- problem solving: 210
- collaboration: 185
- communication: 164

Most Common Tools and Technologies:
- python: 312
- sql: 280
- power bi: 145
```

---

## 💬 Future Scope: Multi-Agent AI System & Conversational Chat

To evolve beyond static batch processing, future iterations introduce autonomous agents and real-time streaming interfaces.

### Autonomous Agent Pipeline (Claude 3.5 Sonnet)

```text
               Unstructured Job Postings (CSV / Web Scraper)
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │        Orchestrator Agent             │
                │        (Powered by Claude)            │
                └──────────────────┬────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   BERT Named    │       │     Skill       │       │ Market Insights │
│  Entity Agent   │       │ Standardization │       │      Agent      │
│ (JobBERT Model) │       │   & Taxonomy    │       │ (Trend & Salary)│
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                ┌───────────────────────────────────────┐
                │   Structured JSON / Vector Database   │
                └───────────────────────────────────────┘
```

### Planned Improvements
1. **ESCO / O*NET Mapping Agents**: Automatically maps extracted skills to international standard taxonomies via tool calling.
2. **Server-Sent Events (SSE) Streaming**: Real-time token delivery to lower UI response latency.
3. **Multi-Turn Context & Gap Analysis**: Interactive chat system where applicants compare their CV directly against target job descriptions.

---

## 🛠 Tech Stack

- **Languages:** Python 3.8+
- **NLP & NER Models:** Hugging Face `transformers`, `jjzha/jobbert_skill_extraction`, `spaCy`
- **Generative AI:** OpenAI API (`gpt-3.5-turbo`), Anthropic API (`claude-3-5-sonnet`)
- **Backend & UI:** FastAPI, Streamlit
