# Procurement Intelligence Assistant

## Dataset Description

This project uses a procurement dataset containing purchase order and supplier information. The dataset includes fields such as:

* PO ID
* Supplier
* Order Date
* Delivery Date
* Item Category
* Order Status
* Quantity
* Unit Price
* Negotiated Price
* Defective Units
* Compliance

Using these columns, additional procurement KPIs are created, including:

* Lead Time Days
* Total Cost
* Negotiated Total Cost
* Cost Savings
* Defect Rate
* Status Score
* Compliance Score
* Risk Score
* Risk Level

## Project Overview

Procurement Intelligence Assistant is an AI-powered assistant that helps users ask natural language questions about procurement data.

It combines:

* SQL Agent for structured data analysis
* RAG for context-based answers
* ChromaDB for vector search
* Groq LLM for reasoning and response generation
* Streamlit for the user interface

### Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

#### Why is the `.env` file used?

- Stores sensitive credentials such as API keys securely without hardcoding them in the source code.
- Prevents accidental exposure of secrets when sharing the project on GitHub or collaborating with others.

#### Important Notes

- Never commit the `.env` file to GitHub. Add it to `.gitignore`.
- Each user must create their own `.env` file and provide a valid Groq API key before running the application.
## How the Project Works

### 1. Data Cleaning

The raw procurement dataset is cleaned by:

* Standardizing column names
* Converting date columns
* Converting numeric columns
* Handling missing values
* Removing duplicate records

### 2. Feature Engineering

New procurement metrics are created:

* Lead Time Days = Delivery Date - Order Date
* Total Cost = Quantity × Unit Price
* Negotiated Total Cost = Quantity × Negotiated Price
* Cost Savings = Total Cost - Negotiated Total Cost
* Defect Rate = Defective Units / Quantity
* Compliance Risk based on Yes/No compliance
* Risk Score using defect rate, lead time, order status, and compliance

### 3. Database Creation

The cleaned dataset is stored in a SQLite database as:

```txt
procurement_orders
```

This allows the SQL Agent to answer analytical questions such as:

```txt
Which supplier has the highest total spend?
How many orders were placed in 2022?
Which category has the highest cost savings?
```

### 4. RAG Pipeline

Each procurement record is converted into a text document.

These documents are embedded using HuggingFace embeddings and stored in ChromaDB.

The RAG system helps answer questions like:

```txt
Explain supplier performance issues.
Summarize procurement risks.
Give recommendations based on supplier data.
```

### 5. Intelligent Routing

The assistant uses an LLM router to decide whether a question should go to:

```txt
SQL
RAG
Hybrid
```

* SQL is used for calculations and exact data queries.
* RAG is used for explanation and context-based answers.
* Hybrid is used when both calculation and explanation are needed.

### 6. Streamlit Interface

The user can ask procurement-related questions through a simple Streamlit interface.

The app also shows execution details, including:

* Route used
* SQL query generated
* SQL result
* Retrieved RAG documents

## Example Questions

```txt
Which supplier has the highest total spend?
How many orders were placed in 2022?
Which supplier has the highest risk score?
Explain procurement risks in the dataset.
Which suppliers have compliance issues?
Summarize supplier performance.
```

## Tech Stack

```txt
Python
Pandas
SQLite
SQLAlchemy
LangChain
Groq LLM
ChromaDB
HuggingFace Embeddings
Streamlit
```

