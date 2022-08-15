import concurrent
from typing import List

import cohere
from tqdm import tqdm


class CohereEmbedder:
    '''Embeds text with the Cohere embedding API'''

    def __init__(self, cohere_api_key: str):
        '''
        Args:
            cohere_api_key: Your Cohere API key
        '''

        self.client = cohere.Client(cohere_api_key)

    def embed(self, texts: List[str], model: str = 'small', shard_size=-1, num_workers=1):
        '''
        Embeds text with the Cohere API.

        **Parameters:**

        * **texts** - a list of strings to embed.
        * **model** - The Cohere API model to use. See the Cohere python client reference.
        * **shard_size** - The number of embeddings to send in each job. If -1, sends one job with all data.
        * **num_workers** - The numbers of parallel embedding jobs to send to the Cohere embedding API

        **Returns:** A list containing an embedding vector for each given text string.
        '''
        if shard_size == -1:
            shard_size == len(texts)
            num_workers = 1
            if num_workers == 1:
                return self.client.embed(model=model, texts=texts).embeddings

        def send_request(i):
            data_shard = texts[i : i + shard_size]
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
