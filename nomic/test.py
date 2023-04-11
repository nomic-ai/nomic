from nomic.gpt4all import GPT4AllGPU
from time import time
LLAMA_PATH = 'decapoda-research/llama-7b-hf'
m = GPT4AllGPU(LLAMA_PATH)
config = {'num_beams': 2,
          'min_new_tokens': 10,
          'max_length': 512,
          'repetition_penalty': 2.0}
t1 = time()
out = m.generate('Write about the theory of relativity', config)
print(out)
t2 = time()
print(f'time taken is: {t2-t1}seconds')
