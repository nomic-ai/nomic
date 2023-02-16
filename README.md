<h1 align="center">Atlas</h1>
<p align="center">Explore, label, search and share massive datasets in your web browser.</p>

<p align="center">
  <a href="https://docs.nomic.ai">:closed_book:	 Documentation</a> 
  <br> <br>
  <br> <br>
</p>

### Example Maps

[:world_map: Twitter](https://atlas.nomic.ai/map/twitter) (5.4 million tweets)

[:world_map: Stable Diffusion](https://atlas.nomic.ai/map/stablediffusion) (6.4 million images)

[:world_map: NeurIPS Proceedings](https://atlas.nomic.ai/map/neurips) (16,623 abstracts)

## Quickstart

Install the Nomic client with:
```bash
pip install nomic
```

Login/create your Nomic account:
```bash
nomic login
```

Follow the instructions to obtain your access token. Enter your access token with:
```bash
nomic login [token]
```

Make your first map:
```python
from nomic import atlas
import numpy as np

num_embeddings = 10000
embeddings = np.random.rand(num_embeddings, 256)

response = atlas.map_embeddings(embeddings=embeddings)
print(response)
```

Explore Atlas' [documentation](https://docs.nomic.ai) to make more advanced maps.
