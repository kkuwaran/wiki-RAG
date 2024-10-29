import os
import pandas as pd
from openai import OpenAI  # openai==1.51.1


# Initialize OpenAI client with API key and base URL
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1",
)

# Define constants for embedding model and output file path
EMBEDDING_MODEL = "text-embedding-3-small"



def generate_text_embedding(text: str, model: str, precision: int = 6) -> list:
    """Generate an embedding for the given text using the specified model."""

    # Clean up text by replacing newline characters
    text = text.replace("\n", " ")

    # Request embedding from the OpenAI API
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding

    # Round embedding values to the specified precision
    embedding = [round(value, precision) for value in embedding]
    return embedding


def embed_dataframe(df: pd.DataFrame, output_file: str, model: str = EMBEDDING_MODEL, 
                    show_preview: bool = True) -> None:
    """Embed text data in the provided DataFrame using the specified model and save it to a CSV file."""

    # Generate embeddings for each row's text data
    df["embedding"] = df["text"].apply(lambda x: generate_text_embedding(x, model=model))

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the DataFrame with embeddings to a CSV file
    df.to_csv(output_file, index=False)

    # Display a preview of the DataFrame if requested
    if show_preview:
        pd.set_option("display.max_colwidth", 150)
        print(df.head(10))

    print(f"\n\n***** Embeddings saved to '{output_file}'.")