import os
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

nlp.max_length = 3000000  # Further increased max_length to handle even larger texts

# Load BERT model and tokenizer for sentence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Neo4j connection setup
uri = "bolt://localhost:7687"
user = "neo4j"
password = "123456789"  # Replace with your actual Neo4j password
driver = GraphDatabase.driver(uri, auth=(user, password))


def close_neo4j_connection(driver):
    """Close the Neo4j database connection."""
    driver.close()


def load_data(file_path):
    """Load text data from the specified file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data


def classify_sentence(sentence):
    """Classify a sentence as causal or non-causal using the BERT model."""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.argmax().item()


def extract_entities(sentence):
    """Extract named entities from a sentence using SpaCy."""
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]


def add_causal_relationship(tx, cause, effect, relationship, company, year):
    """Add a causal relationship to the Neo4j database."""
    query = (
        "MERGE (c:Entity {name: $cause}) "
        "MERGE (e:Entity {name: $effect}) "
        "MERGE (c)-[r:CAUSE_OF {type: $relationship, company: $company, year: $year}]->(e) "
        "RETURN c, e"
    )
    tx.run(query, cause=cause, effect=effect, relationship=relationship, company=company, year=year)


def add_to_graph(driver, cause, effect, relationship, company, year):
    """Wrapper to simplify Neo4j interactions for adding relationships."""
    with driver.session() as session:
        session.write_transaction(add_causal_relationship, cause, effect, relationship, company, year)


def process_chunk(sentences, driver, company, year):
    """Process a chunk of sentences to extract and add causal relationships to Neo4j."""
    for sentence in sentences:
        if classify_sentence(sentence) == 1:  # Identify causal sentence
            entities = extract_entities(sentence)
            if len(entities) >= 2:
                cause, effect = entities[0][0], entities[1][0]
                add_to_graph(driver, cause, effect, "causes", company, year)
                print(f"Added relationship: {cause} -> {effect} (Company: {company}, Year: {year})")


def process_and_add_to_graph(driver, text_chunk, company, year):
    """Process the text chunk by breaking it into manageable chunks and adding relationships to Neo4j."""
    chunk_size = 1000000  # Define a manageable chunk size
    sentences = text_chunk.split(". ")  # Split text into sentences or other manageable units
    current_chunk = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            current_chunk.append(sentence)
            # Check if the current chunk exceeds the max length
            if sum(len(s) for s in current_chunk) > chunk_size:
                process_chunk(current_chunk, driver, company, year)
                current_chunk = []  # Reset the chunk

    # Process the last chunk if it's not empty
    if current_chunk:
        process_chunk(current_chunk, driver, company, year)


def process_company_files(company_path, company_name, driver):
    """Process all files for the specified company."""
    print(f"Processing files for {company_name}")
    available_files = [f for f in os.listdir(company_path) if f.endswith('.txt')]

    for file_name in available_files:
        file_path = os.path.join(company_path, file_name)
        year = file_name.split('-')[0]  # Assuming file names start with the year
        print(f"Processing file: {file_path}")
        data = load_data(file_path)
        process_and_add_to_graph(driver, data, company_name, year)


def main():
    # Dictionary to store company paths
    company_paths = {
        "AAPL": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\AAPL\10-K\0000320193-23-000106",
        "AMZN": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\AMZN\10-K\0001018724-23-000004",
        "GOOGL": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\GOOGL\10-K\0001652044-23-000016",
        "MSFT": r"C:\Users\MJ\OneDrive - Leeds Beckett University\dissertation\sec-edgar-filings\MSFT\10-K\0000789019-23-000014"
    }

    # Print available companies
    print("Available companies:")
    for i, company in enumerate(company_paths.keys(), 1):
        print(f"{i}. {company}")

    # Select company to process
    while True:
        try:
            company_choice = int(input("Enter the number of the company you want to process: ")) - 1
            selected_company = list(company_paths.keys())[company_choice]
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number.")

    # Get the path for the selected company
    company_path = company_paths[selected_company]

    # Process all files for the selected company
    process_company_files(company_path, selected_company, driver)

    # Close the Neo4j connection
    close_neo4j_connection(driver)

    print("\nProcessing complete. You can now visualize the graph in Neo4j Browser.")
    print("Sample Cypher queries:")
    print("1. MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")
    print("2. MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1, r, e2 LIMIT 100")
    print("3. MATCH (cause:Entity)-[:CAUSES]->(effect:Entity) RETURN cause, effect LIMIT 100")
    print("4. MATCH (cause:Entity)-[r:CAUSE_OF]->(effect:Entity) WHERE r.company = 'AAPL' RETURN cause, r, effect LIMIT 50")


if __name__ == "__main__":
    main()
