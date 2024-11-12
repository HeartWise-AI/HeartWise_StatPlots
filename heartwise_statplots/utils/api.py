
import json

def load_api_keys(path: str) -> dict[str, str]:
    """
    Load API keys from a JSON file.

    Args:
        path (str): Path to the JSON file containing the API keys.

    Returns:
        dict[str, str]: A dictionary containing the API keys.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON file.
    """
    try:
        with open(path) as f:
            keys = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The API keys file at {path} does not exist.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"The API keys file at {path} is not a valid JSON file.")
    return keys
