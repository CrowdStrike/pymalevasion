import os
import tempfile

import anthropic
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm import sampling_params as sp


class vllmNemotron:
    def __init__(self, model_path, schema_path=None):
        cache_dir = tempfile.mkdtemp()
        os.environ["OUTLINES_CACHE_DIR"] = cache_dir
        # Set up guided decoding if schema is provided
        self.guided_decoding = None
        if schema_path:
            with open(schema_path, "r") as f:
                schema_json = f.read()
            self.guided_decoding = sp.GuidedDecodingParams(json=schema_json)

        self.llm = LLM(
            model=model_path, tensor_parallel_size=8, enable_chunked_prefill=False, guided_decoding_backend="outlines"
        )

    def ask_chat_model(self, system_prompt, prompts, temperature=0.3, max_tokens=4096, seed=5942):
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, seed=seed, guided_decoding=self.guided_decoding
        )
        prompts = [f"{system_prompt}\n{prompt}" for prompt in prompts]
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        response = [output.outputs[0].text for output in outputs]
        return response


class OpenAIWrapper:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI()

    def ask_chat_model(self, system_prompt, prompts, temperature=0.3, seed=5942):
        responses = []
        for prompt in tqdm(prompts):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=temperature,
                    seed=seed,
                    max_tokens=4096,
                )
                responses.append(completion.choices[0].message.content)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                responses.append("")
        return responses


class AnthropicWrapper:
    def __init__(self, model_name="claude-3-opus-20240229"):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=3)

    def ask_chat_model(self, system_prompt, prompts, temperature=0.3, max_tokens=2000):
        responses = []

        for prompt in tqdm(prompts):
            try:
                response = self.client.messages.create(
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                responses.append(response.content[0].text)

            except Exception as e:
                print(f"Anthropic API error: {e}")
                responses.append("")

        return responses
