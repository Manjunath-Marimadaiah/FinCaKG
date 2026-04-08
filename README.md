# Financial Causality Analysis using Knowledge Graphs

## Overview
This project explores how Natural Language Processing (NLP) and graph databases can be used to extract and analyse cause-and-effect relationships from SEC 10-K filings.

The aim was to build a Financial Causality Knowledge Graph (FinCaKG) that transforms unstructured financial text into a structured graph of entities and relationships. This enables easier analysis of financial risks, strategic actions, and performance drivers across major technology companies.

## Business Problem
Financial reports contain large amounts of unstructured text, making it difficult for analysts to quickly identify the key factors influencing company performance.

Traditional manual analysis is:
- time-consuming
- difficult to scale
- prone to missing important causal relationships

This project addresses that problem by automating the extraction and representation of financial causality from company filings.

## Project Objectives
- Extract causal relationships from SEC 10-K filings
- Classify financial sentences using BERT
- Segment cause and effect components from financial text
- Store extracted relationships in Neo4j as a knowledge graph
- Visualise and analyse the most influential causes and effects

## Dataset
- Source: SEC EDGAR 10-K filings
- Companies included:
  - Apple (AAPL)
  - Amazon (AMZN)
  - Google (GOOGL)
  - Microsoft (MSFT)

## Tools & Technologies
- Python
- BERT / Transformers
- SpaCy
- Neo4j
- Matplotlib
- Google Colab
- ThreadPoolExecutor for parallel processing

## Methodology
The project followed this workflow:

1. Collect SEC 10-K filings from selected companies
2. Clean and preprocess raw financial text
3. Tokenise and extract entities using SpaCy
4. Use BERT to classify causal vs non-causal sentences
5. Segment identified sentences into cause and effect components
6. Store extracted relationships in Neo4j
7. Query and visualise the graph to identify key financial drivers

## Key Features
- Automated extraction of financial cause-effect relationships
- Knowledge graph construction in Neo4j
- Visual exploration of causal links between financial entities
- Analysis of top causes and effects across filings
- Model evaluation using standard classification metrics

## Results
The project demonstrated that NLP and graph-based modelling can improve the analysis of unstructured financial filings.

### Performance Metrics
- Accuracy: 0.89
- Precision: 0.85
- Recall: 0.87
- F1 Score: 0.86

### Key Outcomes
- Built a structured financial causality knowledge graph
- Identified high-frequency causes and effects in company narratives
- Showed how graph-based analysis can support financial risk assessment and strategic decision-making

## Repository Structure
```bash
.
├── README.md
├── pipeline.py
├── data_processing.py
├── requirements.txt
└── images/
    ├── workflow.png
    ├── preprocessing_flow.png
    ├── loss_curve.png
    ├── hyperparameter_heatmap.png
    ├── neo4j_graph.png
    ├── top_causes_effects.png
    └── case_study_graph.png
```
## Visualisations
1. System Workflow

Shows the complete pipeline from SEC filing ingestion to NLP processing and knowledge graph analysis.

<img width="230" height="440" alt="image" src="https://github.com/user-attachments/assets/c595d270-5ecc-4b16-839b-ca83b2903bbc" />

2. Data Preprocessing Workflow

Illustrates text extraction, cleaning, tokenisation, and entity recognition.

<img width="372" height="614" alt="image" src="https://github.com/user-attachments/assets/b105e4e9-227c-4258-8937-9864369cffe5" />

3. Training vs Validation Loss

Shows model learning behaviour during BERT training.

<img width="627" height="361" alt="image" src="https://github.com/user-attachments/assets/de828436-7f7c-42a3-96ff-284d76a1d673" />

4. Hyperparameter Tuning Heatmap

Shows the effect of learning rate and batch size on model performance.

<img width="640" height="441" alt="image" src="https://github.com/user-attachments/assets/c1a24ea0-840d-4b2e-8587-e077a1b49674" />

5. Neo4j Knowledge Graph

Visualises extracted financial entities and cause-effect relationships.

<img width="555" height="499" alt="image" src="https://github.com/user-attachments/assets/13fa5c86-716f-4a7e-b959-dcc77858e847" />

6. Top Causes and Effects

Highlights the most significant causal relationships identified in the filings.

7. Case Study Visualisation

Demonstrates practical use cases such as risk assessment and strategic decision analysis.

## Business Value

This project shows how NLP and knowledge graphs can support:

financial risk analysis
strategic decision support
trend identification
structured exploration of unstructured reports

It can be extended into a decision-support tool for analysts, investors, and finance teams.

## Limitations

Focused on selected technology companies
Primarily designed for English-language filings
Better handling of implicit causality would improve results further
Scalability can still be improved for very large graph sizes
Future Improvements
Fine-tune domain-specific transformer models
Add real-time ingestion of financial news and filings
Build an interactive dashboard on top of Neo4j
Extend to other financial documents such as earnings call transcripts
Improve user interface for non-technical users

## How to Run
Clone the repository

Install dependencies:

pip install -r requirements.txt
Configure Neo4j credentials using environment variables

Run the main pipeline:

python pipeline.py

Note: Sensitive credentials have been removed from this public version of the project. Use your own .env or environment variables when connecting to Neo4j.

## Author

Manjunath Marimadaiah
