import jsonlines
import gym
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)


def parse_query_response(text: str):
    items = [line.strip() for line in text.split("\n") if line.strip()]
    items = [re.sub(r"\d+\.", "", item) for item in items]
    items = [item.strip() for item in items]
    items = list(set(items))
    return items


def calculate_acc(results: dict):
    prec, recall = results["precision"], results["recall"]
    f1 = 2 * prec * recall / (prec + recall + 1e-8)
    return f1


samples_path = "data/query_generation_samples.jsonl"
with open(samples_path) as f:
    samples = list(jsonlines.Reader(f))

for sample in tqdm(samples):
    doc_id = sample["doc_id"]
    response = sample["response"]
    queries = parse_query_response(response)
    sample["parsed"] = queries
    try:
        results = gym.evals.eval_queries(doc_id, queries)
        sample["results"] = results
        sample["reward"] = calculate_acc(results)
    except Exception as e:
        sample["results"] = {}
        reward = 0

with open(samples_path, "w") as f:
    writer = jsonlines.Writer(f)
    writer.write_all(samples)
