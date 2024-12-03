# Import necessary libraries
import os
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from neo4j import GraphDatabase
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Define a custom dataset class for BERT
class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Neo4j connection details
def connect_to_neo4j(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to Neo4j successfully.")
        return driver
    except Exception as e:
        print(f"An error occurred while connecting to Neo4j: {e}")
        return None

# Enhanced entity extraction
def extract_entities(filing_text):
    entities = []
    lines = filing_text.splitlines()
    for line in lines:
        # Detect financial metrics
        if "Revenue" in line:
            entities.append({"name": "Revenue", "type": "Metric"})
        if "Profit" in line:
            entities.append({"name": "Profit", "type": "Metric"})
        if "Operating Income" in line:
            entities.append({"name": "Operating Income", "type": "Metric"})
        
        # Detect products
        if "Azure" in line:
            entities.append({"name": "Azure", "type": "Product"})
        if "Microsoft Teams" in line:
            entities.append({"name": "Microsoft Teams", "type": "Product"})
        if "Xbox" in line:
            entities.append({"name": "Xbox", "type": "Product"})
        
        # Detect new trends
        if "Cloud Computing" in line:
            entities.append({"name": "Cloud Computing", "type": "Trend"})
        if "Artificial Intelligence" in line or "AI" in line:
            entities.append({"name": "Artificial Intelligence", "type": "Trend"})
        if "Sustainability" in line:
            entities.append({"name": "Sustainability", "type": "Trend"})
    return entities

# Enhanced relationship extraction
def extract_relationships(filing_text):
    relationships = []
    lines = filing_text.splitlines()
    for line in lines:
        # Relationships between metrics
        if "Revenue drives Profit" in line:
            relationships.append(("Revenue", "Profit", "CAUSES"))
        if "Expenses reduce Profit" in line:
            relationships.append(("Expenses", "Profit", "DECREASES"))
        
        # Product contributions
        if "Azure contributes to Revenue" in line:
            relationships.append(("Azure", "Revenue", "INCREASES"))
        if "Microsoft Teams impacts Operating Income" in line:
            relationships.append(("Microsoft Teams", "Operating Income", "INCREASES"))
        if "Xbox drives Profit" in line:
            relationships.append(("Xbox", "Profit", "CAUSES"))
        
        # Trends driving products
        if "Cloud Computing drives Azure adoption" in line:
            relationships.append(("Cloud Computing", "Azure", "CAUSES"))
        if "AI powers Microsoft Teams features" in line:
            relationships.append(("Artificial Intelligence", "Microsoft Teams", "ENABLES"))
        
        # Market impacts
        if "Azure increases Market Share" in line:
            relationships.append(("Azure", "Market Share", "INCREASES"))
    return relationships

# Function to create a knowledge graph from filings
def create_knowledge_graph_from_filings(driver, filings):
    if driver is None:
        print("Cannot create graph, Neo4j driver not connected.")
        return

    with driver.session() as session:
        try:
            for filing in filings:
                filing_text = filing['text']
                year = filing['year']

                # Extract entities and relationships from the filing
                entities = extract_entities(filing_text)
                relationships = extract_relationships(filing_text)

                # Create nodes for entities
                for entity in entities:
                    session.run(
                        """
                        MERGE (n:Entity {name: $name})
                        ON CREATE SET n.type = $type, n.year = $year, n.created = timestamp()
                        """,
                        name=entity['name'], type=entity.get('type', 'Unknown'), year=year
                    )

                # Create relationships
                for rel in relationships:
                    session.run(
                        """
                        MATCH (a:Entity {name: $from_name}), (b:Entity {name: $to_name})
                        MERGE (a)-[r:CAUSES {type: $rel_type, created: timestamp()}]->(b)
                        """,
                        from_name=rel[0], to_name=rel[1], rel_type=rel[2]
                    )

            print("Knowledge graph updated successfully with new trends, products, and impacts.")
        except Exception as e:
            print(f"An error occurred while creating the knowledge graph: {e}")

# Load data from nested directory structure
def load_data_from_directory(base_dir):
    data = []
    if not os.path.exists(base_dir):
        print(f"Path does not exist: {base_dir}")
        return data

    for subdir, _, files in os.walk(base_dir):
        for filename in files:
            if filename == 'full-submission.txt':
                file_path = os.path.join(subdir, filename)
                with open(file_path, 'r') as file:
                    text = file.read()
                    if text.strip():
                        year = os.path.basename(subdir)
                        data.append({'text': text, 'year': year, 'label': 0})
                    else:
                        print(f"Empty content in file: {filename} at {subdir}")

    if not data:
        print("No valid data loaded from the directory.")
    return data

# Combine data from multiple years
def combine_data_from_years(base_dir, selected_years):
    combined_data = []
    for year in selected_years:
        year_dir = os.path.join(base_dir, year)
        if not os.path.exists(year_dir):
            print(f"No folder found for the year {year}.")
            continue
        data = load_data_from_directory(year_dir)
        combined_data.extend(data)
    return combined_data

# Plot training and validation loss
def plot_metrics(training_losses, validation_losses):
    """
    Plot the training and validation losses over epochs.
    """
    if not training_losses or not validation_losses:
        print("No data available for plotting losses.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Train BERT model
def train_bert(data, base_dir, epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Train a BERT model using the provided dataset and save the model and tokenizer.
    """
    writer = SummaryWriter(log_dir=os.path.join(base_dir, 'runs/bert_experiment'))

    if len(data) < 2:
        print(f"Not enough data to perform train-test split. Found {len(data)} sample(s).")
        return

    processed_data = preprocess_data(data)
    sentences = [item['text'] for item in processed_data]
    labels = [item['label'] for item in processed_data]

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=512, return_tensors='pt')
    val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=512, return_tensors='pt')

    train_dataset = BertDataset(train_encodings, torch.tensor(train_labels))
    val_dataset = BertDataset(val_encodings, torch.tensor(val_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    use_cuda = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_cuda)

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_cuda:
                with autocast(device_type='cuda'):
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        writer.add_scalar('Training Loss', avg_train_loss, epoch)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        print(f"Epoch {epoch + 1} completed - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the model and tokenizer
    model_save_path = os.path.join(base_dir, "bert_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

    # Plot training and validation losses
    plot_metrics(training_losses, validation_losses)

    writer.close()

# Main function
def main():
    company = 'MSFT'
    print(f"Processing company: {company}")

    base_dir = r'C:\Users\MJ\OneDrive - Leeds Beckett University\data_sets\MSFT\10-K'
    print(f"Searching for files in: {base_dir}")

    available_years = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    print(f"Available years for {company}: {available_years}")
    if not available_years:
        print("No available years found.")
        return

    # Prompt user for year selection
    selected_years = input("Enter the years you want to analyze (comma-separated, e.g., 2018,2019): ").split(',')
    selected_years = [year.strip() for year in selected_years if year.strip() in available_years]

    if not selected_years:
        print("No valid years selected. Exiting.")
        return

    print(f"Processing selected years: {selected_years}")

    combined_data = combine_data_from_years(base_dir, selected_years)
    if not combined_data:
        print("No data to process. Exiting.")
        return

    # Train BERT model (optional)
    train_bert(combined_data, base_dir)

    # Connect to Neo4j
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "1234567890"
    driver = connect_to_neo4j(uri, user, password)

    # Create knowledge graph using extracted data
    create_knowledge_graph_from_filings(driver, combined_data)

    print("\nProcessing complete. You can now visualize the graph in Neo4j Browser.")
    print("Sample Cypher queries:")
    print("\n1. View all nodes:\n\n MATCH (n) RETURN n LIMIT 50;\n")
    print("\n2. View all relationships:\n\n MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50;\n")
    print("\n3. Products contributing to Revenue:\n\n MATCH (n:Product)-[r]->(m:Entity {name: 'Revenue'}) RETURN n, r, m;\n")

if __name__ == "__main__":
    main()
