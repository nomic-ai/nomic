import subprocess
import time
import sys
import random    
from pathlib import Path
# import eliza

prompts = [
    'Write me a letter from the perspective of a cat',
    'Write me a short poem',
    'Tell me how to hard boil an egg',
    'Come up with the vacation destinations.'
]

def download_executable():
    pass

def download_model(modelname):
    pass

executable_path = Path("~/.nomic/gpt4all").expanduser()
model_path = Path("~/.nomic/gpt4all-lora-quantized.bin").expanduser()

class GPT4All():
    def __init__(self, download=True):
        if not (executable_path.exists() and model_path.exists()):
            if not download:
                raise Exception("You need to download the executable and model first.")
            else:
                raise     
        self.bot = subprocess.Popen([executable_path, '--model', model_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # queue up the prompt.
        self._parse_to_prompt(write_to_stdout=False)

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

    def prompt(self, prompt, write_to_stdout = True):
        bot = self.bot
        bot.stdin.write(prompt.encode('utf-8'))
        bot.stdin.write(b"\n")
        bot.stdin.flush()
        return self._parse_to_prompt(write_to_stdout)
    
    def therapy(self):
        session = eliza.Eliza()
        session.load('doctor.txt')
        bot = self.bot
        eliza_says = session.initial()
        print("\n\nELIZA: " + eliza_says)
        while True:
            bot_says = self.prompt(eliza_says)
            eliza_says = session.respond(bot_says)
            print("\n\nELIZA: " + eliza_says)
        
if __name__ == '__main__':
    session = BotSession()
    session.therapy()