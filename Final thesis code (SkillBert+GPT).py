import openai
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import json
import os
import csv
from collections import Counter
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Set your OpenAI API key
openai.api_key = ""

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load BERT model and tokenizer for skill extraction
tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_skill_extraction")
model = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_skill_extraction")
skill_extraction_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Define a set of generic/irrelevant terms to be removed
GENERIC_TERMS = set([
    "identify", "provide", "develop", "ensure", "support", "work", "perform", "collaborate", "manage", "data", "analysis", "business", "reports", "skills", "needs", "complex", "results", "requirements", "systematic"
])

def clean_bert_output(skills):
    """Clean BERT output by filtering irrelevant words, removing stop words, and excluding generic terms."""
    filtered_skills = []
    for skill in skills:
        skill = skill.strip().lower()
        if skill not in STOP_WORDS and skill not in GENERIC_TERMS and skill.isalpha() and len(skill) > 2:
            filtered_skills.append(skill)
    return filtered_skills

def call_openai_api(prompt):
    """Call OpenAI API to refine extracted skills and remove noise."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that refines and enhances skill extraction from job descriptions. You should extract and list technical skills, non-technical skills, and tools/technologies accurately."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        result_text = response.choices[0].message['content'].strip()
        return result_text
    except Exception as e:
        print(f"Error during OpenAI API request: {e}")
        return ""

def parse_openai_response(response):
    """Parse OpenAI response to extract skills and tools/technologies."""
    lines = response.split('\n')
    technical_keywords = []
    non_technical_keywords = []
    tools_and_technologies = []

    parsing_technical = False
    parsing_non_technical = False
    parsing_tools = False

    for line in lines:
        line = line.strip()

        if line.lower().startswith('technical skills:'):
            parsing_technical = True
            parsing_non_technical = False
            parsing_tools = False
        elif line.lower().startswith('non-technical skills:'):
            parsing_technical = False
            parsing_non_technical = True
            parsing_tools = False
        elif line.lower().startswith('tools and technologies:'):
            parsing_technical = False
            parsing_non_technical = False
            parsing_tools = True
        else:
            if parsing_technical and line:
                technical_keywords.append(line.strip('-').strip())
            elif parsing_non_technical and line:
                non_technical_keywords.append(line.strip('-').strip())
            elif parsing_tools and line:
                tools_and_technologies.append(line.strip('-').strip())

    return technical_keywords, non_technical_keywords, tools_and_technologies

def get_skills_for_job_title(job_title, job_description):
    """Extract technical skills, non-technical skills, and tools/technologies from job description using multiple methods."""
    
    # Tokenize and truncate job description for BERT
    try:
        max_length = min(tokenizer.model_max_length, 512)  # Safe maximum length
    except AttributeError:
        max_length = 512  # Default if model_max_length is not available
    try:
        tokens = tokenizer(job_description, truncation=True, max_length=max_length, return_tensors='pt')
        job_description_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        job_description_text = job_description

    # Extract skills using JobBERT model
    try:
        skill_extraction_results = skill_extraction_pipeline(job_description_text)
        technical_keywords = [result['word'] for result in skill_extraction_results if result['entity'].startswith('B')]
        non_technical_keywords = [result['word'] for result in skill_extraction_results if result['entity'].startswith('I')]

        # Clean the extracted keywords
        technical_keywords = clean_bert_output(technical_keywords)
        non_technical_keywords = clean_bert_output(non_technical_keywords)

    except Exception as e:
        print(f"Error during skill extraction pipeline execution: {e}")
        technical_keywords, non_technical_keywords = [], []

    # Prepare OpenAI prompt
    combined_skills_text = (
        f"Extracted Technical Skills:\n{', '.join(technical_keywords) if technical_keywords else 'None'}\n"
        f"Extracted Non-Technical Skills:\n{', '.join(non_technical_keywords) if non_technical_keywords else 'None'}\n"
    )

    prompt = (
        f"Refine and enhance the following extracted skills and identify any additional tools/technologies. "
        f"Make sure to include all relevant technical and non-technical skills:\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"{combined_skills_text}\n"
        f"Tools and Technologies:\n"
    )

    # Call OpenAI API for refinement
    openai_response = call_openai_api(prompt)

    # Parse OpenAI API response
    additional_technical_keywords, additional_non_technical_keywords, tools_and_technologies = parse_openai_response(openai_response)

    # Combine BERT and OpenAI results (display both)
    combined_technical_keywords = list(set(technical_keywords + additional_technical_keywords))
    combined_non_technical_keywords = list(set(non_technical_keywords + additional_non_technical_keywords))

    # Return BERT and OpenAI skills
    return {
        'technical_keywords': combined_technical_keywords,
        'non_technical_keywords': combined_non_technical_keywords,
        'tools_and_technologies': tools_and_technologies
    }

def main():
    # Path to your CSV file
    csv_file_path = 'F:/April 2024 Backup This PC/Downloads April 2024/Data_Analyst_Indeed_Jobs.csv'

    # Initialize counters for skills across all job descriptions
    technical_skills_counter = Counter()
    non_technical_skills_counter = Counter()
    tools_counter = Counter()

    # Open the CSV file and process all job titles and descriptions
    with open(csv_file_path, 'r', encoding='latin-1') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            job_title = row['Job Title']
            job_description = row['Job Description'].lower()

            # Get BERT and OpenAI refined results
            results = get_skills_for_job_title(job_title, job_description)

            # Update counters with the results
            technical_skills_counter.update(results['technical_keywords'])
            non_technical_skills_counter.update(results['non_technical_keywords'])
            tools_counter.update(results['tools_and_technologies'])

    # Display the most common technical skills, non-technical skills, and tools/technologies
    print("\nMost Common Technical Skills:")
    for skill, count in technical_skills_counter.most_common(10):
        print(f"{skill}: {count}")

    print("\nMost Common Non-Technical Skills:")
    for skill, count in non_technical_skills_counter.most_common(10):
        print(f"{skill}: {count}")

    print("\nMost Common Tools and Technologies:")
    for tool, count in tools_counter.most_common(10):
        print(f"{tool}: {count}")

if __name__ == '__main__':
    main()
