"""
Generate samples of query sets for literature review titles
"""

import os

cwd = os.getcwd()
os.sys.path.append(cwd)

from models.inference import ModelInference
from models import prompts
import gym
import re

# base_mistral_model = "mistralai/Mistral-7B-v0.1"
ft_mistral_model = "mistralai/Mistral-7B-Instruct-v0.1"
max_tokens = 200
inference = ModelInference(
    model_path=ft_mistral_model,
)

def parse_list(text: str):
    items = [line.strip() for line in text.split("\n") if line.strip()]
    items = [re.sub(r"\d+\.", "", item) for item in items]
    items = [item.strip() for item in items]
    items = list(set(items))
    return items


# Load titles
doc_ids = gym.db.get_doc_ids_with_citations()
print(len(doc_ids))
for doc_id in doc_ids:
    print(doc_id)
    title = gym.db.get_document(doc_id)["title"]
    print(title)
    prompt = f"Generate a set of short keyword queries to find papers related to the following title: {title}\n\nQueries:\n\n"
    print(prompt)
    response = inference.generate_response(
        prompt=prompt,
        max_tokens=max_tokens,
    )
    response = response.split("Queries:")[1].strip()
    print("Generated Response:")
    print(response)
    query_set = parse_list(response)
    print("Query Set:")
    print(query_set)
    # Evaluate
    query_set = [f'"{query}"' for query in query_set]
    results = gym.evals.eval_queries(doc_id, query_set)
    print("Evaluation Results:")
    print(results)
    print("")
    input()

