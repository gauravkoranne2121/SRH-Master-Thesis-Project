import csv
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the spaCy model for keyword extraction
nlp = spacy.load('en_core_web_sm')

# Define common technical skills
PREDEFINED_TECHNICAL_SKILLS = {
    'python', 'java', 'c++', 'sql', 'javascript', 'html', 'css',
    'aws', 'docker', 'kubernetes', 'tensorflow', 'pytorch', 'hadoop',
    'spark', 'excel', 'tableau', 'powerbi', 'sas', 'r', 'bigquery'
}

# Initialize the JobBERT model and tokenizer
def initialize_jobbert():
    try:
        tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_knowledge_extraction")
        model = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_knowledge_extraction")
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        print(f"Error initializing JobBERT model: {e}")
        raise

pipe = initialize_jobbert()

def extract_entities(text):
    """Extract entities from text using the JobBERT model."""
    try:
        results = pipe(text)
        entities = [result['word'].lower() for result in results if result['entity'].startswith('B-')]
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []

def remove_stopwords_and_get_keywords(text):
    doc = nlp(text.lower())
    
    technical_keywords = set()
    non_technical_keywords = set()

    for token in doc:
        if token.is_alpha and token.text not in STOP_WORDS:
            if token.pos_ in ['NOUN', 'PROPN']:
                if token.text in PREDEFINED_TECHNICAL_SKILLS:
                    technical_keywords.add(token.text)
                else:
                    non_technical_keywords.add(token.text)
            elif token.pos_ == 'ADJ':
                non_technical_keywords.add(token.text)
    
    return list(technical_keywords), list(non_technical_keywords)

def get_skills_for_job_description(job_description):
    """Returns technical and non-technical keywords associated with the job description."""
    technical_keywords, non_technical_keywords = remove_stopwords_and_get_keywords(job_description)
    
    # If keywords not found, use JobBERT model
    if not technical_keywords and not non_technical_keywords:
        entities = extract_entities(job_description)
        # Assume entities are technical for simplicity
        technical_keywords.extend(entities)
    
    return list(set(technical_keywords)), list(set(non_technical_keywords))

# Function to calculate precision, recall, and F1 score
def calculate_precision_recall_f1(extracted_skills, ground_truth_skills):
    extracted_skills_set = set(extracted_skills)
    ground_truth_set = set(ground_truth_skills)

    true_positives = len(extracted_skills_set & ground_truth_set)
    false_positives = len(extracted_skills_set - ground_truth_set)
    false_negatives = len(ground_truth_set - extracted_skills_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def main():
    try:
        # Ground truth skills for evaluation (replace with actual data for testing)
        ground_truth_skills_dict = {
            'Data Analyst': ['sql', 'excel', 'tableau', 'python', 'data analysis'],
            'Machine Learning Engineer': ['python', 'tensorflow', 'pytorch', 'machine learning', 'deep learning'],
            # Add more job titles and corresponding ground truth skills here
        }

        # Initialize overall skills sets
        overall_technical_skills = set()
        overall_non_technical_skills = set()

        # Read CSV and process job descriptions
        with open('F:\/April 2024 Backup This PC/Downloads April 2024/Data_Analyst_Indeed_Jobs.csv', 'r', encoding='latin-1') as csvfile:
            reader = csv.DictReader(csvfile)

            overall_precision = 0
            overall_recall = 0
            overall_f1 = 0
            job_count = 0
            all_extracted_technical_skills = set()  # Accumulate all technical skills extracted
            all_ground_truth_skills = set()  # Accumulate all ground truth skills
            
            for row in reader:
                job_title = row['Job Title']
                job_description = row['Job Description']
                
                # Get extracted skills from the job description
                technical_keywords, non_technical_keywords = get_skills_for_job_description(job_description)

                # Update overall skills
                overall_technical_skills.update(technical_keywords)
                overall_non_technical_skills.update(non_technical_keywords)

                # Use the ground truth for comparison if available
                if job_title in ground_truth_skills_dict:
                    ground_truth_skills = ground_truth_skills_dict[job_title]

                    # Accumulate all extracted and ground truth skills for overall accuracy
                    all_extracted_technical_skills.update(technical_keywords)
                    all_ground_truth_skills.update(ground_truth_skills)

                    # Calculate precision, recall, and F1 for this job
                    precision, recall, f1 = calculate_precision_recall_f1(technical_keywords, ground_truth_skills)
                    
                    # Accumulate precision, recall, and F1
                    overall_precision += precision
                    overall_recall += recall
                    overall_f1 += f1
                    job_count += 1

            # Calculate overall averages for precision, recall, and F1
            if job_count > 0:
                overall_precision /= job_count
                overall_recall /= job_count
                overall_f1 /= job_count

                # Calculate overall accuracy for all jobs combined
                final_precision, final_recall, final_f1 = calculate_precision_recall_f1(list(all_extracted_technical_skills), list(all_ground_truth_skills))

                # Print overall skills
                print(f"\nOverall Technical Skills: {', '.join(sorted(overall_technical_skills))}")
                print(f"Overall Non-Technical Skills: {', '.join(sorted(overall_non_technical_skills))}")


                # Print overall accuracy across all job descriptions combined
                print(f"\nFinal Precision for all jobs combined: {final_precision:.2f}")
                print(f"Final Recall for all jobs combined: {final_recall:.2f}")
                print(f"Final F1 Score for all jobs combined: {final_f1:.2f}")

    except FileNotFoundError:
        print("CSV file not found. Please check the file path.")
        return

if __name__ == "__main__":
    main()
