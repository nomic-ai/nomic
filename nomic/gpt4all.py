import os
import sys
import time
import torch
import random
import requests
import subprocess
from tqdm import tqdm
from pathlib import Path
from loguru import logger

class GPT4AllGPU():
    def __init__(self, alpaca_path=None):
        if alpaca_path is None:
            raise ValueError('Please pass a path to your alpaca model.')

        self.model_path = alpaca_path
        self.tokenizer_path = alpaca_path
        self.lora_path = 'nomic-ai/vicuna-lora-multi-turn_epoch_2'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        added_tokens = tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})

        if added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))

        self.model = PeftModelForCausalLM.from_pretrained(self.model,
                                                          self.lora_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16)
        self.model.to(dtype=torch.float16)
        print(f"Mem needed: {self.model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")


class GPT4All():
    def __init__(self, force_download=False):
        self.executable_path = Path("~/.nomic/gpt4all").expanduser()
        self.model_path = Path("~/.nomic/gpt4all-lora-quantized.bin").expanduser()
        if force_download or not (self.executable_path.exists() or self.model_path.exists()):
            logger.info('Downloading model...')
            self._download()
        self.bot = subprocess.Popen([executable_path, '--model', model_path],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
        # queue up the prompt.
        self.parse_to_prompt()

    def _download(self):
        response = requests.get('https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized.bin',
                                stream=True)
        if response.status_code == 200:
            os.makedirs(self.executable_path, exist_ok=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(self.model_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size // 8192, unit='KB'):
                    f.write(chunk)
            print(f"File downloaded successfully to {target_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

    def parse_to_prompt(self, write_to_stdout = True):
        bot_says = ['']
        point = b''
        bot = self.bot
        while True:
            point += bot.stdout.read(1)
            try:
                character = point.decode("utf-8")
                if character == "\f":
                    return "\n".join(bot_says)
                if character == "\n":
                    bot_says.append('')
                    sys.stdout.write('\n')
                else:
                    bot_says[-1] += character
                    sys.stdout.write(character)
                    sys.stdout.flush()
                point = b''

            except UnicodeDecodeError:
                if len(point) > 4:
                    point = b''

    def prompt(self, response):
        bot = self.bot
        bot.stdin.write(response.encode('utf-8'))
        bot.stdin.write(b"\n")
        bot.stdin.flush()
        return self.parse_to_prompt()
    

if __name__ == '__main__':
    model = GPT4All(force_download=True)