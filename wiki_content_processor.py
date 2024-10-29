import requests
import pandas as pd
from dateutil.parser import parse


def fetch_wikipedia_content(title: str = '2024', verbose: bool = True) -> str:
    """Fetches plaintext content for a given Wikipedia page title using the MediaWiki API."""
    
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exlimit": 1,
        "titles": title,
        "explaintext": True,
        "formatversion": 2,
        "format": "json"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    content = response.json()["query"]["pages"][0]["extract"]

    # Preview content
    if verbose:
        chars = 2500
        print(f"Previewing first {chars} characters of Wikipedia content: \n")
        print(content[:chars], "...\n\n")  
    return content


def clean_content(content: str, verbose: bool = True) -> pd.DataFrame:
    """Cleans and processes Wikipedia content into a DataFrame with formatted event descriptions."""

    # Split by newline and create DataFrame
    lines = content.split("\n")
    df = pd.DataFrame(lines, columns=["text"])
    
    # Remove empty lines and headings (except "Nobel Prizes")
    df = df[~df["text"].str.contains(r"^\s*$")]
    df = df[~df["text"].str.contains("^==.*==$") | df["text"].str.contains("Nobel Prizes")]

    # Display cleaned DataFrame
    if verbose:
        print("Cleaned DataFrame:")
        pd.set_option("display.max_colwidth", 200)
        n_display = 30
        print(df.head(n_display))
        print("\n")
        print(df.tail(n_display))
    
    return df


def process_events(df: pd.DataFrame, year: str = '2024') -> pd.DataFrame:
    """Processes each line in the DataFrame to format dates and events into a standardized format."""

    processed_df = pd.DataFrame(columns=["text"])
    prefix = f"{year} Overview"

    for _, row in df.iterrows():
        text: str = row["text"]

        # Skip lines that are section headings
        if text.startswith("== "):
            continue

        # Case I: Line is (only) a date or an undated event
        elif " – " not in text[:20]:
            try:
                # Case I.A: Line is a date
                parse(text)
                prefix = text + f', {year}'  # Save date as prefix
            except ValueError:
                # Case I.B: Line is an event (without a date)
                processed_df = add_event(processed_df, prefix, text)

        # Case II: Line is an event with a date or Nobel Prizes field
        else:
            # Extract prefix and attempt parsing as date
            cur_prefix, text_no_prefix = text.split(" – ", 1)
            # Handle cases where the cur_prefix is a range of dates
            cur_prefix_2 = cur_prefix.split("–")[0]
            try:
                # Case II.A: Line is an event with a date
                parse(cur_prefix_2)  # Check if it's a date
                prefix = cur_prefix + f', {year}'  # Save date or duration as prefix
                processed_df = add_event(processed_df, prefix, text_no_prefix)
            except ValueError:
                # Case II.B: Field is likely related to Nobel Prizes
                field_of_study, text = text.split(" – ", 1)
                prefix = f"{year} Nobel Prizes in {field_of_study}"
                processed_df = add_event(processed_df, prefix, text)
                
    return processed_df


def add_event(df: pd.DataFrame, prefix: str, event_text: str) -> pd.DataFrame:
    """Appends a new row with a formatted event to the DataFrame."""

    new_row = {"text": f"{prefix} – {event_text}"}
    concat_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return concat_df


def fetch_and_process_wikipedia_content(title: str, verbose: bool = True) -> pd.DataFrame:
    """Fetches and processes Wikipedia content into a DataFrame with formatted event descriptions."""

    # Fetch Wikipedia content and clean it
    content = fetch_wikipedia_content(title, verbose)
    df = clean_content(content, verbose)
    processed_df = process_events(df, title)

    # Display processed DataFrame
    if verbose:
        print("Processed DataFrame:")
        pd.set_option("display.max_colwidth", 200)
        n_display = 60
        print(processed_df.head(n_display))
        print("\n")
        print(processed_df.tail(n_display))

    return processed_df