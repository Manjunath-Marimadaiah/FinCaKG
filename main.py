import os
import re
import torch
import spacy
import warnings
from transformers import BertTokenizer, BertForSequenceClassification
from neo4j import GraphDatabase

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load SpaCy model for Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

nlp.max_length = 2000000  # Increase the maximum length limit for large documents

# Load BERT model and tokenizer for sentence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Neo4j connection setup
uri = "bolt://localhost:7687"
user = "neo4j"
password = "123456789"  # Update with your actual Neo4j password
driver = GraphDatabase.driver(uri, auth=(user, password))

def close_neo4j_connection(driver):
    """Close the Neo4j connection."""
    driver.close()

def load_data(file_path):
    """Load data from a text file."""
    print(f"Trying to open file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def classify_sentence(sentence):
    """Classify a sentence as causal or non-causal using BERT."""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.argmax().item()

def extract_entities(sentence):
    """Extract named entities from a sentence using SpaCy."""
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]

def add_causal_relationship(tx, cause, effect, relationship):
    """Add a causal relationship to the Neo4j database."""
    query = (
        "MERGE (c:Entity {name: $cause}) "
        "MERGE (e:Entity {name: $effect}) "
        "MERGE (c)-[r:CAUSE_OF {type: $relationship}]->(e) "
        "RETURN c, e"
    )
    tx.run(query, cause=cause, effect=effect, relationship=relationship)

def add_to_graph(driver, cause, effect, relationship):
    """Add a causal relationship to the Neo4j graph."""
    with driver.session() as session:
        session.write_transaction(add_causal_relationship, cause, effect, relationship)

def process_and_add_to_graph(driver, text_chunk):
    """Process a text chunk to extract and add causal relationships to the graph."""
    sentences = text_chunk.split(". ")
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and classify_sentence(sentence) == 1:  # If the sentence is classified as causal
            entities = extract_entities(sentence)
            if len(entities) >= 2:
                cause, effect = entities[0][0], entities[1][0]
                add_to_graph(driver, cause, effect, "causes")
                print(f"Added relationship: {cause} -> {effect}")

def process_company_files(base_dir, company_name, driver):
    """Process all files for a given company."""
    chunk_size = 1000000
    print(f"Processing file: {base_dir}")
    data = load_data(base_dir)

    for i in range(0, len(data), chunk_size):
        text_chunk = data[i:i + chunk_size]
        process_and_add_to_graph(driver, text_chunk)

def main():
    """Main function to process the company data and visualize in Neo4j."""
    company_paths = {
        "AAPL": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\AAPL\10-K\0000320193-23-000106\full-submission.txt",

        "AMZN": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\AMZN\10-K\0001018724-23-000004\full-submission.txt",
        "GOOGL": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\GOOGL\10-K\0001652044-23-000016\full-submission.txt",
        "MSFT": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\MSFT\10-K\0000789019-23-000014\full-submission.txt"
    }

    for company, path in company_paths.items():
        try:
            print(f"Processing data for {company}")
            process_company_files(path, company, driver)
        except FileNotFoundError as e:
            print(e)  # Log the error and continue with the next company

    close_neo4j_connection(driver)
    print("Processing complete. Visualize your graph in the Neo4j Browser.")

if __name__ == "__main__":
    main()
