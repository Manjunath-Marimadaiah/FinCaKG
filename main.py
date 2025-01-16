# Import necessary libraries
import os  # Used for interacting with the file system
import re  # Used for regular expression operations
import torch  # Used for PyTorch operations
from torch.utils.data import DataLoader, Dataset  # For data handling and loading
from transformers import BertForSequenceClassification, BertTokenizer  # Used for BERT model and tokenizer
from torch.optim import AdamW  # Used for optimizer
from sklearn.model_selection import train_test_split  # Used for splitting datasets
from torch.amp import autocast, GradScaler  # For mixed-precision training
from torch.utils.tensorboard import SummaryWriter  # Used for logging and visualization
import matplotlib.pyplot as plt  # For plotting graphs
from neo4j import GraphDatabase  # Used for connecting to Neo4j
import spacy  # Used for natural language processing
import json  # Used for working with JSON data
import logging  # For logging messages
from typing import List, Dict, Optional  # For type hinting

# Used to configure logging for debugging and status updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Used to set up logging format
logger = logging.getLogger(__name__)  # Used to create a logger instance

# Used to load SpaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")  # Load the small English NLP model for named entity recognition

# Define a custom dataset class for BERT
class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  #Used to store tokenized inputs
        self.labels = labels  # Used to store corresponding labels

    def __len__(self):
        return len(self.labels)  # Return the number of samples

    def __getitem__(self, idx):
        # Used to retrieve a single data point and its label
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]  # Include labels with the item
        return item

# Used to function to connect to Neo4j database
def connect_to_neo4j(uri: str, user: str, password: str):
    try:
        # Used to establish a connection to the Neo4j database
        driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j successfully.")  # Log successful connection to Neo4j
        return driver
    except Exception as e:
        logger.error(f"An error occurred while connecting to Neo4j: {e}")  # Log connection errors
        return None  # Return None if the connection fails

# Enhanced entity extraction with chunking
def extract_entities(filing_text: str, max_length: int = 1000000) -> List[Dict[str, str]]:
    entities = []  # Used to initialize an empty list to store entities
    if len(filing_text) > max_length:
        logger.info(f"Text length {len(filing_text)} exceeds the maximum of {max_length}. Splitting into chunks...")
        for i in range(0, len(filing_text), max_length):
            # can be Used to split the text into manageable chunks
            chunk = filing_text[i:i + max_length]
            doc = nlp(chunk)  # Process each chunk using SpaCy
            # Extract entities from the chunk and append to the list
            entities.extend([{"name": ent.text, "type": ent.label_} for ent in doc.ents])
    else:
        # Used to process the entire text if within the maximum length
        doc = nlp(filing_text)
        # Used to extract entities from the text and append to the list
        entities.extend([{"name": ent.text, "type": ent.label_} for ent in doc.ents])
    logger.info(f"Extracted {len(entities)} entities.")  # Log the number of entities extracted
    return entities

# Used to enhanced relationship extraction
def extract_relationships(filing_text: str) -> List[tuple]:
    relationships = []  # Initialize an empty list to store relationships
    lines = filing_text.splitlines()  # Split the text into lines for easier processing

    # Used to log the first 10 lines of text for debugging
    logger.info(f"First 10 lines of the text:\n{lines[:10]}")

    # Extract corporate address using a regular expression
    if "Address" in filing_text:
        address_pattern = r"Address:\s*(.*?)\s*City"  # Used to define a regex pattern for the address
        match = re.search(address_pattern, filing_text)  # Used to search for the pattern in the text
        if match:
            relationships.append(("Microsoft Corp", match.group(1), "HAS_ADDRESS"))  # Add the address relationship

    # Identify products and services offered
    product_keywords = ["software", "cloud", "devices", "services"]  # Used to define product keywords
    for keyword in product_keywords:
        if keyword in filing_text:
            relationships.append(("Microsoft Corp", keyword.capitalize(), "OFFERS"))  # Add product relationships

    # Extract strategic focuses
    strategy_keywords = ["cloud-first", "intelligent cloud", "digital transformation"]  # Define strategy keywords
    for strategy in strategy_keywords:
        if strategy in filing_text:
            relationships.append(("Microsoft Corp", strategy, "FOCUSES_ON"))  # Used to add strategy relationships

    # Associate risk factors
    if "Risk Factors" in filing_text:
        relationships.append(("Risk", "Microsoft Corp", "ASSOCIATED_WITH"))  # Used to add risk factor relationship

    logger.info(f"Extracted {len(relationships)} relationships.")  # Log the number of relationships extracted
    return relationships

# Save entities and relationships as JSON for debugging
def save_to_json(data: List[Dict], filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # Used to save data to a JSON file with indentation for readability
    logger.info(f"Saved data to {filename}")  # Used to log the save operation

# Function to create a knowledge graph from filings
def create_knowledge_graph_from_filings(driver, filings: List[Dict]):
    if driver is None:
        logger.error("Cannot create graph, Neo4j driver not connected.")  # Log error if Neo4j connection is not established
        return

    with driver.session() as session:
        try:
            all_entities = []  # Used to initialize a list to store all entities
            all_relationships = []  # Used to initialize a list to store all relationships
            for filing in filings:
                filing_text = filing['text']  # Extract text from the filing
                year = filing['year']  # Extract year from the filing
                # Used to extract entities and relationships from the filing text
                entities = extract_entities(filing_text)
                relationships = extract_relationships(filing_text)
                all_entities.extend(entities)  # Used to append extracted entities to the list
                all_relationships.extend(relationships)  # Used to append extracted relationships to the list

                # Add entities to Neo4j
                for entity in entities:
                    session.run(
                        """
                        MERGE (n:Entity {name: $name})
                        ON CREATE SET n.type = $type, n.year = $year, n.created = timestamp()
                        """,
                        name=entity['name'], type=entity.get('type', 'Unknown'), year=year
                    )

                # Used to add relationships to Neo4j
                for rel in relationships:
                    session.run(
                        """
                        MERGE (a:Entity {name: $from_name})
                        MERGE (b:Entity {name: $to_name})
                        MERGE (a)-[r:`""" + rel[2] + """`]->(b)
                        ON CREATE SET r.created = timestamp()
                        """,
                        from_name=rel[0], to_name=rel[1], rel_type=rel[2]
                    )

            # Save extracted data for further analysis
            save_to_json(all_entities, "entities.json")
            save_to_json(all_relationships, "relationships.json")
            logger.info("Knowledge graph updated successfully.")  # Log success message
        except Exception as e:
            logger.error(f"An error occurred: {e}")  # Used to log any errors encountered

# This section processes the knowledge graph with relationships to create a structure that can be further used for graph neural networks.
# Function to load data from a directory
def load_data_from_directory(base_dir: str) -> List[Dict[str, str]]:
    data = []  # Initialize a list to store data
    for subdir, _, files in os.walk(base_dir):
        for filename in files:
            if filename == 'full-submission.txt':  # Used to look for files named 'full-submission.txt'
                file_path = os.path.join(subdir, filename)  # Used to construct file path
                with open(file_path, 'r') as file:
                    text = file.read().strip()  # Read and clean the text
                    year = os.path.basename(subdir)  # Extract the year from the directory name
                    data.append({'text': text, 'year': year})  # Used to append the data to the list
    return data

# Function to combine data for selected years
def combine_data_from_years(base_dir: str, selected_years: List[str]) -> List[Dict[str, str]]:
    combined_data = []  # Initialize a list to store combined data
    for year in selected_years:
        year_dir = os.path.join(base_dir, year)  # Used to construct directory path for the year
        if os.path.exists(year_dir):
            year_data = load_data_from_directory(year_dir)  # Load data for the year
            combined_data.extend(year_data)  # Used to append the data to the combined list
        else:
            logger.warning(f"Directory for year {year} not found. Skipping...")  # Log warning if directory not found
    return combined_data

# Function to plot loss curves
def plot_loss_curves(training_losses: List[float], validation_losses: List[float]):
    plt.figure(figsize=(10, 6))  # Used to Create a new figure
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o')  # Plot training losses
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss', marker='o')  # Plot validation losses
    plt.xlabel('Epochs')  # Used to label x-axis
    plt.ylabel('Loss')  # Used to label y-axis
    plt.title('Training and Validation Loss Over Epochs')  # Set title
    plt.legend()  # Used to add legend
    plt.grid(True)  # Used to enable grid
    plt.show()  # Used to display the plot

# Graph neural networks can be applied here by taking this structure to perform node classification or edge prediction.
# Function to train BERT model with hyperparameter tuning
def train_bert(data: List[Dict[str, str]], base_dir: str, epochs: int = 30, batch_size: int = 8, learning_rate: float = 5e-5):
    writer = SummaryWriter(log_dir=os.path.join(base_dir, 'runs/bert_experiment'))  # Set up TensorBoard logging
    sentences = [item['text'] for item in data]  # Used to extract sentences from data
    labels = [0] * len(sentences)  # Used to assign example labels (modify as needed)

    # Split data into training and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    # Tokenize sentences for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Load pre-trained BERT tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')  # Load pre-trained BERT model
    train_encodings = tokenizer(train_sentences, truncation=True, padding=True, max_length=512, return_tensors='pt')  # Used to tokenize training data
    val_encodings = tokenizer(val_sentences, truncation=True, padding=True, max_length=512, return_tensors='pt')  # Used to tokenize validation data

    # Create dataset and dataloader
    train_dataset = BertDataset(train_encodings, torch.tensor(train_labels))  # Create dataset for training
    val_dataset = BertDataset(val_encodings, torch.tensor(val_labels))  # Used to create dataset for validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Used to create dataloader for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Create dataloader for validation

    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Used to set up optimizer
    use_cuda = torch.cuda.is_available()  # Used to check if GPU is available
    scaler = GradScaler(enabled=use_cuda)  # Set up mixed precision training if GPU is available

    training_losses = []  # Initialize list to store training losses
    validation_losses = []  # Used to initialize list to store validation losses

    for epoch in range(epochs):
        model.train()  # Used to set model to training mode
        total_train_loss = 0  # Used to initialize total training loss

        for batch in train_loader:
            optimizer.zero_grad()  # Clear gradients
            if use_cuda:
                with autocast(device_type='cuda'):
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])  # Forward pass
                    loss = outputs.loss  # Used to compute loss
                scaler.scale(loss).backward()  # Used to backward pass with scaled gradients
                scaler.step(optimizer)  # Update weights
                scaler.update()  # Used to update scaler
            else:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])  # Forward pass
                loss = outputs.loss  # Used to compute loss
                loss.backward()  # Used for backward pass
                optimizer.step()  # Update weights

            total_train_loss += loss.item()  # Used to accumulate training loss

        avg_train_loss = total_train_loss / len(train_loader)  # Compute average training loss
        training_losses.append(avg_train_loss)  # Used to append to training losses
        writer.add_scalar('Training Loss', avg_train_loss, epoch)  # Log training loss to TensorBoard

        model.eval()  # Used to set model to evaluation mode
        total_val_loss = 0  # Used to initialize total validation loss

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])  # Forward pass
                loss = outputs.loss  # Used to compute loss
                total_val_loss += loss.item()  # Accumulate validation loss

        avg_val_loss = total_val_loss / len(val_loader)  # Used to compute average validation loss
        validation_losses.append(avg_val_loss)  # Append to validation losses
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)  # Used to log validation loss to TensorBoard
        logger.info(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")  # Log epoch progress

    plot_loss_curves(training_losses, validation_losses)  # Used to plot training and validation losses
    writer.close()  # Close TensorBoard writer

# Main function
def main():
    company = 'MSFT'  # Company symbol
    base_dir = r'C:\Users\MJ\OneDrive - Leeds Beckett University\data_sets\MSFT\10-K'  # Base directory for filings

    # List available years of data
    available_years = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]  # Used to get list of years from directories
    logger.info(f"Available years for {company}: {available_years}")  # Used to log available years

    # Prompt user for selected years
    selected_years = input("Enter the years to analyze (comma-separated, e.g., 2018,2019): ").split(',')  # Get input from user
    selected_years = [year.strip() for year in selected_years if year.strip() in available_years]  # Filter valid years

    if not selected_years:
        logger.error("No valid years selected. Exiting...")  # Log error if no valid years
        return

    # Combine data from selected years
    combined_data = combine_data_from_years(base_dir, selected_years)  # Combine data for selected years
    if not combined_data:
        logger.error("No data available for the selected years. Exiting...")  # Used to log error if no data available
        return

    # Train the BERT model
    train_bert(combined_data, base_dir, epochs=30)  # Train the BERT model with combined data

    # Connect to Neo4j and create the knowledge graph
    uri = "bolt://localhost:7687"  # Neo4j URI
    user = "neo4j"  # Neo4j username
    password = "1234567890"  # Neo4j password
    driver = connect_to_neo4j(uri, user, password)  # Used to connect to Neo4j database

    # The Neo4j graph structure can now be used as input to graph neural networks for tasks like link prediction and node classification.
    # Create the knowledge graph from filings
    create_knowledge_graph_from_filings(driver, combined_data)  # Build the knowledge graph

    # Log sample Cypher queries for Neo4j visualization
    logger.info("\nProcessing complete. You can now visualize the graph in Neo4j Browser.")  # Used to log completion
    logger.info("Sample Cypher queries:")
    logger.info("\n1. View all nodes:\n\n MATCH (n) RETURN n LIMIT 50;\n")  # sample query to view all nodes
    logger.info("\n2. View all relationships:\n\n MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50;\n")  # Sample query to view all relationships
    
if __name__ == "__main__":
    main() # used to call the main() function
