import argparse
import json
import os
import pathlib

from jinja2 import Template


def parse_args():
    parser = argparse.ArgumentParser(description="LLM calls for processing scripts")
    parser.add_argument("--model", help="Name of the model or ID")
    parser.add_argument("--scripts-path", help="Directory containing scripts to analyze")
    parser.add_argument("--output-name", default="results.jsonl", help="Name of the output JSONL file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument(
        "--prompt-template-path", default="templates/malicious_check.j2", help="Path to the Jinja2 template file"
    )
    parser.add_argument("--system-prompt-template-path", default="templates/default_system_prompt.j2")
    return parser.parse_args()


def load_template(template_path):
    with open(template_path, "r") as f:
        return Template(f.read())


def main():
    args = parse_args()
    if args.model == "gpt-4o":
        from llm_utils import OpenAIWrapper

        model = OpenAIWrapper(args.model)
    elif "claude" in args.model:
        from llm_utils import AnthropicWrapper

        model = AnthropicWrapper(args.model)
    else:
        from llm_utils import vllmNemotron

        model = vllmNemotron(args.model)

    # Setup paths
    output_dir = "results/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output_name)

    # Clear output file if it exists
    open(output_file, "w").close()

    # Load template
    template = load_template(args.prompt_template_path)
    system_prompt = load_template(args.system_prompt_template_path)
    system_prompt = system_prompt.render()

    prompts = []
    filenames = []

    for filename in os.listdir(args.scripts_path):
        file_path = os.path.join(args.scripts_path, filename)
        script = pathlib.Path(file_path).read_text(encoding="utf-8", errors="replace")
        message = template.render(script=script)

        prompts.append(message)
        filenames.append(filename)

        if len(prompts) == args.batch_size or filename == os.listdir(args.scripts_path)[-1]:
            try:
                responses = model.ask_chat_model(system_prompt, prompts)

                for idx, response in enumerate(responses):
                    result = {"filename": filenames[idx]}
                    result["raw_response"] = response

                    # Append result to file
                    with open(output_file, "a") as f:
                        f.write(json.dumps(result) + "\n")
                        f.flush()

            except Exception as e:
                print(f"Error processing batch: {str(e)}")

            # Clear batch
            prompts = []
            filenames = []


if __name__ == "__main__":
    main()
