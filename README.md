# SRH-Master-Thesis-Project
# Hybrid NLP Pipeline for Automated Skill & Technology Extraction

> **Master's Thesis Research Project**  
> An automated NLP framework combining Transformer-based Named Entity Recognition (JobBERT) and Large Language Models (OpenAI GPT) to extract, categorize, and quantify technical skills, soft skills, and tools from unstructured job postings.

---

## 📌 Abstract / Overview

Accurate extraction and categorization of skill requirements from job descriptions are critical for labor market analysis, automated candidate matching, and educational planning. This project implements a **hybrid Natural Language Processing (NLP) framework**:

1. **Initial NER Extraction**: Utilizes `jjzha/jobbert_skill_extraction` (a fine-tuned BERT architecture) alongside `spaCy` to identify token-level entities in job text.
2. **Rule-Based Post-Processing**: Filters out generic stop words, low-frequency tokens, and noise.
3. **LLM Refinement Layer**: Leverages OpenAI's GPT models (`gpt-3.5-turbo`) to contextualize, enhance, and categorize entities into three distinct buckets:
   - **Technical Skills**
   - **Non-Technical / Soft Skills**
   - **Tools & Technologies**
4. **Aggregated Analytics**: Aggregates output across large-scale datasets (e.g., Indeed job postings CSV) to surface top industry demand indicators.

---

## 🏗 System Architecture

```text
 Unstructured Job Postings (CSV)
                │
                ▼
  [ spaCy Tokenization & Truncation ]
                │
                ▼
  [ JobBERT Model (Token Classification) ] ──► Extracted B/I Entity Tags
                │
                ▼
  [ Rule-Based Text Cleaning ] ──► Removes Stop Words & Generic Terms
                │
                ▼
  [ OpenAI GPT Refinement Engine ] ──► Contextual Disambiguation
                │
                ▼
  [ Aggregation & Categorization ] ──► Frequency Distribution Output
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An active [OpenAI API Key](https://platform.openai.com/)

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configuration

Set your OpenAI API key in your environment variables or directly inside the script:

```bash
# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY="your-api-key-here"
```

### 3. Data Preparation

Place your job descriptions CSV file in the project directory. Ensure the CSV contains at least the following columns:
- `Job Title`
- `Job Description`

Update the `csv_file_path` variable in `main()` to point to your local dataset path.

### 4. Running the Pipeline

```bash
python main.py
```

---

## 📊 Sample Output

After processing the dataset, the pipeline outputs aggregated counts for the most common entities across all job descriptions:

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

## 🛠 Tech Stack

- **Primary Language:** Python 3.8+
- **Transformers / Models:** Hugging Face (`transformers`), `jjzha/jobbert_skill_extraction`
- **NLP Libraries:** `spaCy`
- **Generative AI / LLM:** OpenAI API (`gpt-3.5-turbo`)
- **Data Manipulation:** `csv`, `collections.Counter`

---

## 📄 Citation & Attribution

If you use this work or model in your research, please attribute the JobBERT model creators:

* JobBERT model by **jjzha**: [`jjzha/jobbert_skill_extraction`](https://huggingface.co/jjzha/jobbert_skill_extraction)
