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
