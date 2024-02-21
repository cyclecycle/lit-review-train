"""
Generate samples of query sets for literature review titles
"""

import jsonlines
import os

cwd = os.getcwd()
os.sys.path.append(cwd)

from inference import ModelInference
import gym
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

base_mistral_model = "mistralai/Mistral-7B-Instruct-v0.2"
max_tokens = 512
inference = ModelInference(
    model_path=base_mistral_model,
)


logging.info("Loading documents...")
doc_ids = gym.db.get_doc_ids_with_citations()
logging.info(f"Loaded {len(doc_ids)} document IDs")

samples_path = "data/query_generation_samples.jsonl"

skip_existing = True
if skip_existing and os.path.exists(samples_path):
    logging.info("Skipping already done samples...")
    with open(samples_path) as f:
        samples_already = list(jsonlines.Reader(f))
    sample_ids_already = [sample["doc_id"] for sample in samples_already]
    doc_ids = [doc_id for doc_id in doc_ids if doc_id not in sample_ids_already]

doc_ids = doc_ids[:10]
logging.info(f"Loading {len(doc_ids)} documents...")
docs = [gym.db.get_document(doc_id) for doc_id in doc_ids]

samples_per_title = 2
logging.info("Generating samples...")
samples = []
for doc in tqdm(docs):
    doc_id, title = doc["doc_id"], doc["title"]
    logging.info(f"Title: {title}, Doc ID: {doc_id}")
    prompt = f"Generate a set of short keyword queries to find papers related to the following title: {title}\n\nQueries:\n\n"
    for _ in range(samples_per_title):
        print(f"Prompt: {prompt}")
        response = inference.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.9,
        )
        response = response.split(prompt)[-1]
        print(f"Response: \n{response}")
        samples.append(
            {
                "task": "query_generation",
                "doc_id": doc_id,
                "title": title,
                "prompt": prompt,
                "response": response,
            }
        )

# Save results
with open(samples_path, "a") as f:
    writer = jsonlines.Writer(f)
    writer.write_all(samples)
    writer.close()
