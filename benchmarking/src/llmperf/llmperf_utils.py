import json
import time
from collections.abc import Iterable
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import tiktoken

NUM_RNG_ATTEMPTS = 10  # Unlikely to be used in practice: prevents eternal WHILE-loops

# Model family identifiers for prompt templating
FAMILY_MODEL_TYPE_IDENTIFIER = {
    'mistral': ['mistral'],
    'llama2': ['llama2'],
    'llama3': ['llama3'],
    'llama4': ['llama4'],
    'deepseek': ['deepseek'],
    'qwen': ['qwen', 'qwq'],
}

LVLM_IMAGE_PATHS = {
    'small': './imgs/vision_perf_eval-small.jpg',
    'medium': './imgs/vision_perf_eval-medium.jpg',
    'large': './imgs/vision_perf_eval-large.jpg',
}


class TiktokenWrapper:
    """Wrapper around tiktoken to provide a consistent interface for token counting.

    Uses cl100k_base encoding which works well for most modern LLMs.
    For benchmarking purposes, exact token counts don't need to match the model's
    tokenizer perfectly - we just need consistent counting for performance metrics.
    """

    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding('cl100k_base')
        self.pad_token: Optional[str] = None  # tiktoken doesn't have pad tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into string tokens (for compatibility with transformers API)."""
        token_ids = self.encoding.encode(text)
        return [self.encoding.decode([tid]) for tid in token_ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of string tokens back to a string."""
        return ''.join(tokens)


class LLMPerfResults:
    """Class with LLM Performance results"""

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata['timestamp'] = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Updates and flattens dictionary

        Returns:
            dict: transformed dictionary
        """
        data = {
            'name': self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self) -> str:
        """Transforms dictionary to json string

        Returns:
            str: json string
        """
        data = self.to_dict()
        return json.dumps(data)


def find_family_model_type(model_name: str) -> str:
    """Finds family model type for prompt templating.

    Args:
        model_name (str): model name

    Returns:
        str: family model type
    """
    for family, models in FAMILY_MODEL_TYPE_IDENTIFIER.items():
        for model in models:
            if model in model_name.lower().replace('-', '').replace('v', ''):
                return family
    return 'llama3'  # Default to llama3 for most modern models


def get_tokenizer(model_name: str) -> TiktokenWrapper:
    """Gets a lightweight tokenizer for token counting.

    Uses tiktoken (cl100k_base encoding) which is much lighter than
    HuggingFace transformers and works well for benchmarking purposes.

    Args:
        model_name (str): model name (ignored, uses cl100k_base for all models)

    Returns:
        TiktokenWrapper: tokenizer wrapper with encode() method
    """
    # For benchmarking, we use a single tokenizer for all models.
    # The exact token count doesn't need to match perfectly -
    # we just need consistent counting for performance metrics.
    return TiktokenWrapper()


def flatten(item: Union[Iterable[Union[str, Iterable[str]]], str]) -> Generator[str, None, None]:
    """Flattens an iterable"""
    for sub_item in item:
        if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
            yield from flatten(sub_item)
        else:
            yield sub_item


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flattens dictionary

    Args:
        d (dict): input dictionary
        parent_key (str, optional): parent key. Defaults to "".
        sep (str, optional): separator. Defaults to "_".

    Returns:
        dict: output flat dictionary
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
