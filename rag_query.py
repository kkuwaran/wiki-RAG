import os
import time
import ast
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import distance
from openai import OpenAI
import tiktoken


# Initialize OpenAI client with API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1",
)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_KWARGS = {
    'model': 'gpt-4o',
    'max_input_tokens': 500,
    'max_output_tokens': 150,
    'temperature': 0.7,
    'top_p': 0.9,
}
INSTRUCTION = """
Please answer the question using the context provided below. 
If the context does not contain sufficient information 
to answer the question, respond with 'I don't know.'
"""



def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list:
    """Generate an embedding for a given text using the specified model."""

    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def calculate_cosine_distance(query: str, file_names: List[str], model: str = EMBEDDING_MODEL,
                              embedding_col: str = "embedding", verbose: bool = True) -> pd.DataFrame:
    """Calculate cosine distance between query and entries in an embedding file."""

    dfs = list()
    columns = list()
    for file_name in file_names:
        # Load and parse embeddings
        assert os.path.exists(file_name), f"File {file_name} not found"
        assert file_name.endswith(".csv"), "Only CSV files are supported"
        df = pd.read_csv(file_name)

        # Check if columns are consistent across files
        if len(columns) == 0:
            columns = df.columns
        else:
            assert (columns == df.columns).all(), "Columns in all dataframes should be the same"
        # Check if embedding column is present
        assert embedding_col in df.columns, f"{embedding_col} column not found in the dataframe"

        # Convert string representation of list to array
        df[embedding_col] = df[embedding_col].apply(ast.literal_eval).apply(np.array)

        # Compute cosine distance to query embedding
        query_embedding = get_embedding(query, model)
        df["cosine_distance"] = df[embedding_col].apply(lambda x: distance.cosine(x, query_embedding))

        # Append dataframe to list
        dfs.append(df)

    # Concatenate dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Sort results by proximity to query
    df = df.sort_values("cosine_distance", ascending=True, ignore_index=True)
    if verbose:
        print(f"Top {min(10, len(df))} closest matches to the query '{query}':")
        print(df.head(10), end="\n\n")
    return df


def create_prompt(instruction: str, context_text: str, query: str) -> str:
    """Constructs a prompt template for RAG."""

    return f"""
Instruction: {instruction} 
Context: \n{context_text}\n
Question: {query}\n
Answer:"""


def construct_final_prompt(df: pd.DataFrame, query: str, tokenizer: tiktoken.core.Encoding, 
                           max_input_tokens: int, instruction: str = INSTRUCTION,
                           verbose: bool = True) -> str:
    """Build a prompt with combined contexts under a token limit for the LLM."""

    contexts, final_prompt = [], ""
    for _, row in df.iterrows():
        # Add new context text and test token limit
        contexts.append(row["text"])
        long_context = "\n".join(contexts)
        prompt = create_prompt(instruction, long_context, query)
        
        # Check token count
        n_tokens = len(tokenizer.encode(prompt))
        if verbose:
            print(f"Total contexts = {len(contexts)}: Total tokens = {n_tokens}")
        if n_tokens > max_input_tokens:
            break

    # Build final prompt with contexts that fit within the limit
    final_long_context = "\n".join(contexts[:-1])
    final_prompt = create_prompt(instruction, final_long_context, query)
    if verbose:
        print("\n=========================================")
        print(f"Final prompt with {len(contexts)-1} contexts: \n{final_prompt}")
        print("=========================================\n\n")
    return final_prompt


def query_openai(prompt: str, model: str, max_tokens: int, 
                 temperature: float, top_p: float) -> str:
    """Query the OpenAI API using a constructed prompt."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    time.sleep(5)  # Avoid rate limit errors
    r = response.choices[0].message.content.strip().strip("\n")
    return r


def get_answer(query: str, file_names: List[str], embedding_model: str = EMBEDDING_MODEL,
               embedding_col: str = "embedding", llm_kwargs: dict = LLM_KWARGS, verbose: bool = True) -> str:
    """Retrieves an answer to a query using RAG by combining relevant contexts with LLM."""

    # Extract model and tokenization parameters
    model = llm_kwargs.get('model', LLM_KWARGS['model'])
    max_input_tokens = llm_kwargs.get('max_input_tokens', LLM_KWARGS['max_input_tokens'])
    max_output_tokens = llm_kwargs.get('max_output_tokens', LLM_KWARGS['max_output_tokens'])
    temperature = llm_kwargs.get('temperature', LLM_KWARGS['temperature'])
    top_p = llm_kwargs.get('top_p', LLM_KWARGS['top_p'])

    # Calculate cosine distances
    df = calculate_cosine_distance(query, file_names, embedding_model, embedding_col, verbose)

    # Set up tokenizer
    tokenizer = tiktoken.encoding_for_model(model)

    # Construct final prompt and query OpenAI API
    final_prompt = construct_final_prompt(df, query, tokenizer, max_input_tokens, instruction=INSTRUCTION, verbose=verbose)
    answer = query_openai(final_prompt, model, max_output_tokens, temperature, top_p)
    return answer