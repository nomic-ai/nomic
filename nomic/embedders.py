import cohere
import concurrent
from tqdm import tqdm
from typing import List

class CohereEmbedder:

    def __init__(self, cohere_api_key: str):

        self.client = cohere.Client(cohere_api_key)


    def embed(self, texts: List[str], model: str = 'large', shard_size=-1, num_workers=1):


        if shard_size == -1:
            shard_size == len(texts)
            num_workers = 1

        def send_request(i):
            data_shard = texts[i: i+shard_size]
            response = self.client.embed(model=model, texts=data_shard)
            return response

        responses = {}
        with tqdm(total=len(texts) // shard_size) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(send_request, i): i for i in range(0, len(texts), shard_size)}
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    responses[int(futures[future])] = response.embeddings
                    pbar.update(1)

        embeddings = []
        for embedding_shard in sorted(list(responses.keys())):
            embeddings += responses[embedding_shard]

        return embeddings



