import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def save_rankings(results, output_path):
    dtype = [
        ("filename", "U64"),
        ("top5", "U64", (5,)),  # Array of 5 strings
        ("scores", "f4", (5,)),  # Array of 5 float32 scores
    ]

    data = np.zeros(len(results), dtype=dtype)
    for i, (key, value) in enumerate(results.items()):
        data[i]["filename"] = key
        data[i]["top5"] = value["top5"]
        data[i]["scores"] = value["score"]

    np.save(output_path, data)


class CodeSimilarityAnalyzer:
    def __init__(self, model_name="intfloat/e5-small-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Compute average pooling"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def compute_embedding(self, code_string):
        """Compute embedding for a single code string"""
        # Add prefix for E5 model
        code_string = f"query: {code_string}"

        inputs = self.tokenizer(code_string, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().numpy()

    def process_file(self, file_path):
        """Read and compute embedding for a single file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.compute_embedding(content)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def analyze_corpus(self, metadata_path, scripts_base_path, output_path=None):
        """Main analysis pipeline"""
        # Read metadata
        df = pd.read_csv(metadata_path)

        # Process train and test sets
        df_train = df[(df["subset"] == "train") & (df["label"] == "clean")]
        df_test = df[(df["subset"] == "test") & (df["label"] == "dirty")]

        print(f"Processing {len(df_train)} train samples and {len(df_test)} test samples")

        # Compute embeddings only for needed samples
        embeddings = {}

        # Process train samples
        print("Processing train samples...")
        for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
            file_path = Path(scripts_base_path) / f"{row['sha256']}.py"
            if file_path.exists():
                embedding = self.process_file(file_path)
                if embedding is not None:
                    embeddings[row["sha256"]] = embedding

        # Process test samples
        print("Processing test samples...")
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            file_path = Path(scripts_base_path) / f"{row['sha256']}.py"
            if file_path.exists():
                embedding = self.process_file(file_path)
                if embedding is not None:
                    embeddings[row["sha256"]] = embedding

        # Convert to arrays for similarity computation
        train_files = [sha for sha in df_train["sha256"] if sha in embeddings]
        test_files = [sha for sha in df_test["sha256"] if sha in embeddings]

        print(f"Successfully processed {len(train_files)} train and {len(test_files)} test files")

        # Create embedding arrays
        embds_train = np.array([embeddings[sha] for sha in train_files])
        embds_test = np.array([embeddings[sha] for sha in test_files])

        # Compute similarities (multiply by 100 to match E5 scale)
        sims = np.matmul(embds_test, embds_train.T) * 100
        rankings = np.argsort(-sims)

        # Create results dictionary
        results = {}
        for ind, test_sha in enumerate(test_files):
            index_rank = rankings[ind, :5]
            scores = sims[ind, index_rank]
            results[test_sha] = {"top5": [train_files[i] for i in rankings[ind, :5]], "score": scores.tolist()}

        # Save results if output path provided
        if output_path:
            save_rankings(results, output_path)

        return results, embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Code Similarity Analysis using E5 Embeddings")
    parser.add_argument("--model-name", default="intfloat/e5-small-v2", help="Name of the model to use for embeddings")
    parser.add_argument("--metadata-path", required=True, help="Path to the corpus metadata CSV file")
    parser.add_argument("--scripts-path", required=True, help="Base path to the scripts directory")
    parser.add_argument("--output-path", required=True, help="Path to save the results")
    return parser.parse_args()


def main():
    args = parse_args()

    analyzer = CodeSimilarityAnalyzer()

    results, embeddings = analyzer.analyze_corpus(
        metadata_path=args.metadata_path, scripts_base_path=args.scripts_path, output_path=args.output_path
    )


if __name__ == "__main__":
    main()
