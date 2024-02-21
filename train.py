"""
Fine-tune a pre-trained language model via reinforcement learning to generate queries in order to maximise precision and recall for a given review title.
"""

import json
import os

cwd = os.getcwd()

samples_path = "data/query_generation_samples.json"
with open(samples_path) as f:
    samples = json.load(f)