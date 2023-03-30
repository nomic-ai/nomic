import os
import sys
import requests
import subprocess
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import platform
try:
    import torch
except ImportError:
    torch = None
    pass

class GPT4AllGPU():
    def __init__(self, llama_path=None):
        from peft import PeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if llama_path is None:
            raise ValueError('Please pass a path to your alpaca model.')

        self.model_path = llama_path
        self.tokenizer_path = llama_path
        self.lora_path = 'nomic-ai/vicuna-lora-multi-turn_epoch_2'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        added_tokens = self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})

        if added_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = PeftModelForCausalLM.from_pretrained(self.model,
                                                          self.lora_path,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16)
        self.model.to(dtype=torch.float16)
        print(f"Mem needed: {self.model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")

    def generate(self, prompt, generate_config=None):
        if generate_config is None:
            generate_config = {}

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=input_ids,
                                      **generate_config)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return decoded[len(prompt):]


def prompt(prompt, model = 'gpt4all-lora-quantized'):
    with GPT4All(model) as gpt4all:
        return gpt4all.prompt(prompt)
    
class GPT4All():
    def __init__(self, model = 'gpt4all-lora-quantized', force_download=False, decoder_config=None):
        """
        :param model: The model to use. Currently supported are 'gpt4all-lora-quantized' and 'gpt4all-lora-unfiltered-quantized'
        :param force_download: If True, will overwrite the model and executable even if they already exist. Please don't do this!
        :param decoder_config: Default None. A dictionary of key value pairs to be passed to the decoder

        """
        if decoder_config is None:
            decoder_config = {}

        self.bot = None
        self.model = model
        self.decoder_config = decoder_config
        assert model in ['gpt4all-lora-quantized', 'gpt4all-lora-unfiltered-quantized']
        self.executable_path = Path("~/.nomic/gpt4all").expanduser()
        self.model_path = Path(f"~/.nomic/{model}.bin").expanduser()

        if force_download or not self.executable_path.exists():
            logger.info('Downloading executable...')
            self._download_executable()
        if force_download or not (self.model_path.exists() and self.model_path.stat().st_size > 0):                                   
            logger.info('Downloading model...')
            self._download_model()

    def __enter__(self):
        self.open()
        return self
    
    def open(self):
        if self.bot is not None:
            self.close()
        # This is so dumb, but today is not the day I learn C++.
        creation_args = [self.executable_path, '--model', self.model_path]
        for k, v in self.decoder_config.items():
            creation_args.append(f"--{k}")
            creation_args.append(str(v))
        
        self.bot = subprocess.Popen(creation_args,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)

        # queue up the prompt.
        self._parse_to_prompt(write_to_stdout=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug("Ending session...")
        self.close()

    def close(self):
        self.bot.kill()
        self.bot = None

    def _download_executable(self):
        if not self.executable_path.exists():
            plat = platform.platform()
            if 'macOS' in plat and 'arm64' in plat:
                upstream = 'https://static.nomic.ai/gpt4all/gpt4all-pywrap-mac-arm64'
            elif 'Linux' in plat:
                upstream = 'https://static.nomic.ai/gpt4all/gpt4all-pywrap-linux-x86_64'
            else:
                raise NotImplementedError(f"Your platform is not supported: {plat}. Current binaries supported are x86 Linux and ARM Macs.")
            response = requests.get(upstream, stream=True)
            if response.status_code == 200:
                os.makedirs(self.executable_path.parent, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(self.executable_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size // 8192, unit='KB'):
                        f.write(chunk)
                self.executable_path.chmod(0o755)                
                print(f"File downloaded successfully to {self.executable_path}")

            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

    def _download_model(self):
        # First download the quantized model.

        if not self.model_path.exists():
            response = requests.get(f'https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/{self.model}.bin',
                                    stream=True)
            if response.status_code == 200:
                os.makedirs(self.model_path.parent, exist_ok=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(self.model_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size // 8192, unit='KB'):
                        f.write(chunk)
                print(f"File downloaded successfully to {self.model_path}")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

    def _parse_to_prompt(self, write_to_stdout = True):
        bot_says = ['']
        point = b''
        bot = self.bot
        while True:
            point += bot.stdout.read(1)
            try:
                character = point.decode("utf-8")
                if character == "\f": # We've replaced the delimiter character with this.
                    return "\n".join(bot_says)
                if character == "\n":
                    bot_says.append('')
                    if write_to_stdout:
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                else:
                    bot_says[-1] += character
                    if write_to_stdout:
                        sys.stdout.write(character)
                        sys.stdout.flush()
                point = b''

            except UnicodeDecodeError:
                if len(point) > 4:
                    point = b''

    def prompt(self, prompt, write_to_stdout = False):
        """
        Write a prompt to the bot and return the response.
        """
        bot = self.bot
        continuous_session = self.bot is not None
        if not continuous_session:
            logger.warning("Running one-off session. For continuous sessions, use a context manager: `with GPT4All() as bot: bot.prompt('a'), etc.`")
            self.open()
        bot.stdin.write(prompt.encode('utf-8'))
        bot.stdin.write(b"\n")
        bot.stdin.flush()
        return_value = self._parse_to_prompt(write_to_stdout)
        if not continuous_session:
            self.close()
        return return_value        

