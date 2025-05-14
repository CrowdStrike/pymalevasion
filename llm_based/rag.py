import argparse
import json
import os
from pathlib import Path

import numpy as np
from jinja2 import Template
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="RAG-based script modification using LLMs")
    parser.add_argument("--model", default="gpt-4o", help="Name of the model to use")
    parser.add_argument("--scripts-path", required=True, help="Directory containing scripts to analyze")
    parser.add_argument("--rankings-path", required=True, help="Path to the rankings npy file")
    parser.add_argument("--output-name", default="results.jsonl", help="Directory for output files")
    parser.add_argument("--system-prompt-path", help="Path to the system prompt file")
    parser.add_argument("--examples-count", type=int, default=3, help="Number of examples to use for each query")
    parser.add_argument("--user-prompt-path", default="templates/rag_user_prompt.j2")
    return parser.parse_args()


def load_rankings(rankings_path):
    data = np.load(rankings_path, allow_pickle=False)
    # Convert to dictionary format for compatibility
    return {str(row["filename"]): {"top5": list(row["top5"]), "score": list(row["scores"])} for row in data}


def load_template(template_path):
    with open(template_path, "r") as f:
        return Template(f.read())


def create_message(script, examples, template):
    examples_formatted = [f"<example>\n{ex}\n</example>" for ex in examples]
    examples_message = "\n".join(examples_formatted)
    return template.render(examples=examples_message, script=script)


def main():
    args = parse_args()
    user_prompt = load_template(args.user_prompt_path)

    # Initialize model based on argument
    if args.model == "gpt-4o":
        from llm_utils import OpenAIWrapper

        model = OpenAIWrapper(args.model)
    elif "claude" in args.model:
        from llm_utils import AnthropicWrapper

        model = AnthropicWrapper(args.model)
    else:
        from llm_utils import vllmNemotron

        model = vllmNemotron(args.model)

    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_name)

    system_prompt = load_template(args.system_prompt_path)
    system_prompt = system_prompt.render()

    test_dict = load_rankings(args.rankings_path)

    # Storage for results
    actions = []
    raw_modifications = {}

    # Process files
    for file in tqdm(test_dict):
        try:
            # Read script with line numbers
            script_path = Path(args.scripts_path) / f"{file}.py"
            script_lines = script_path.read_text().splitlines()
            script = "".join(f"{idx + 1}    {line}\n" for idx, line in enumerate(script_lines))

            # Get example contents
            example_contents = []
            for example_file in test_dict[file]["top5"][: args.examples_count]:
                example_path = Path(args.scripts_path) / f"{example_file}.py"
                example_contents.append(example_path.read_text())

            # Prepare message and get model response
            message = create_message(script, example_contents, user_prompt)
            response = model.ask_chat_model(system_prompt, [message])[0]
            result = {"filename": file}
            result["raw_response"] = response

            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue


if __name__ == "__main__":
    main()
