from pathlib import Path

from nomic.gpt4all import GPT4All


def main():
    model = GPT4All()
    print(model.model, model.model_path)
    model = GPT4All(model_path="/somewhere/thats/not/right")
    print(model.model, model.model_path)
    model.model_path = Path(f"/media/safe/dl/llama/bin/{model.model}.bin")
    print(model.model, model.model_path)
    model.open()
    print(model.prompt("1 + 1 ="))


if __name__ == '__main__':
    main()
