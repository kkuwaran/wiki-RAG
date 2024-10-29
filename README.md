# wiki-RAG: Retrieval-Augmented Generation with Wikipedia Content Embeddings
Wiki-RAG is a Python-based project designed to retrieve answers from Wikipedia using embeddings of selected articles. This project primarily focuses on key events from 2023 and 2024 and allows users to ask questions based on these embeddings, bypassing the need for fetching or embedding the data again.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that combines embedding-based similarity search with a language model to generate answers. Wikipedia articles on events of 2023 and 2024 were embedded beforehand and stored as `.csv` files to avoid redundancy. Users can query these embeddings to retrieve relevant information and generate answers.

### Key Components

- **Embedding Generation**: The `embedding_generator.py` and `wiki_content_processor.py` scripts handle the retrieval and embedding of Wikipedia content. This step is optional for users since embeddings are already provided in the `database` folder.
- **Question Answering**: The `rag_query.py` script enables querying the embeddings to find the best-matching context and generates a response using a language model.

## Project Structure

```plaintext
wiki-RAG/
├── main.ipynb                    # Main notebook for user interaction
├── wiki_content_processor.py     # Fetch and process Wikipedia content
├── embedding_generator.py        # Generate embeddings for the content
├── rag_query.py                  # Query the RAG model to generate answers
│
├── database/                     
│   ├── embedded_wiki_2023.csv    # Precomputed embeddings for 2023 events
│   └── embedded_wiki_2024.csv    # Precomputed embeddings for 2024 events
│
├── requirements.txt              # Required Python packages for the project
└── README.md                     # Project documentation
```

## Getting Started

### Prerequisites

1. **Python 3.7 or higher**
2. **Python Packages**: Install required packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up an OpenAI API key:
   * Obtain an API key from OpenAI.
   * Add it to your environment variables
     ```bash
     export OPENAI_API_KEY='your_api_key'
     ```
   * **Note:** The current code uses a `base_url` set to `"https://openai.vocareum.com/v1"`. If you’re using a standard OpenAI API, you might need to change this to `"https://api.openai.com/v1"` in `embedding_generator.py` and `rag_query.py`.
4. Files
   * **database/embedded_wiki_2023.csv** and **database/embedded_wiki_2024.csv**: These files contain precomputed embeddings for the Wikipedia articles on 2023 and 2024 events.
  
## Usage

1. Open `main.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells in **Part 2** only to ask questions based on the embeddings provided.
   * **Part 1: Fetch and Embed Wikipedia Content** can be skipped since the embeddings are already included.

### Example Workflow

1. Ask a Question:
   * Define your question in `main.ipynb`:
     ```bash
     user_query = "What are the important events in Thailand?"
     ```
   * Set model parameters if needed:
     ```bash
     max_input_tokens = 400
     max_output_tokens = 200
     ```
2. Get an Answer:
   * Run the query in Part 2 of the notebook, which will use the precomputed embeddings and language model to return an answer.



