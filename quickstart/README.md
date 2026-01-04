<a href="https://www.infercom.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/light-logo.png" height="100">
  <img alt="Infercom logo" src="../images/dark-logo.png" height="100">
</picture>
</a>


# Infercom API QuickStart Guide

This guide walks through setting up an API key, performing a few sample queries with and without LangChain, and shares example applications to bootstrap application development for common AI use cases. Let's get started!

## Setting up Infercom API Key

1. Create an account on the [Infercom Portal](https://cloud.infercom.ai/) to get an API key.
2. Once logged in, navigate to the API section and generate a new key.
3. Set your API key as an environment variable:
   ```shell
   export INFERCOM_API_KEY="your-api-key-here"
   ```

## Supported Models

Infercom currently supports the following models: `gpt-oss-120b`, `Meta-Llama-3.3-70B-Instruct`, `DeepSeek-V3-0324-cb`.

## Query the API

Install the OpenAI Python library:
```shell
pip install openai
```

Perform a chat completion:

```python
from openai import OpenAI
import os

api_key = os.environ.get("INFERCOM_API_KEY")

client = OpenAI(
    base_url="https://api.infercom.ai/v1/",
    api_key=api_key,
)

model = "Meta-Llama-3.3-70B-Instruct"
prompt = "Tell me a joke about artificial intelligence."

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    stream=True,
)

response = ""
for chunk in completion:
    response += chunk.choices[0].delta.content or ""

print(response)
```

## Using Infercom APIs with Langchain

Install `langchain-sambanova`:
```shell
pip install -U langchain-sambanova
```

Here's an example of using Infercom's APIs with the Langchain library:

```python
import os
from langchain_sambanova import ChatSambaNova

api_key = os.environ.get("INFERCOM_API_KEY")

llm = ChatSambaNova(
    api_key=api_key,
    streaming=True,
    model="Meta-Llama-3.3-70B-Instruct",
)

response = llm.invoke('What is the capital of France?')
print(response.content)
```

This code snippet demonstrates how to set up a Langchain `ChatSambaNova` instance with Infercom's APIs, specifying the API key, streaming option, and model. You can then use the `llm` object to generate completions by passing in prompts.

## Get Help

- For Infercom-specific support, contact support@infercom.ai (for registered users)
- For technical questions about the underlying SambaNova technology, visit the [SambaNova Community](https://community.sambanova.ai)
- Create an issue on [GitHub](https://github.com/InfercomAI/ai-starter-kit/issues) for bugs or feature requests
- More inference models and features are coming soon!


## Contribute

Building something cool? We welcome contributions! If you have ideas for new quickstart projects or improvements to existing ones, please [open an issue](https://github.com/InfercomAI/ai-starter-kit/issues/new) or submit a [pull request](https://github.com/InfercomAI/ai-starter-kit/pulls) and we'll respond right away.
