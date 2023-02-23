<h1 align="center">Atlas</h1>
<p align="center">Explore, label, search and share massive datasets in your web browser.</p>

<div align="center">
  <a href="https://colab.research.google.com/drive/1bquOLIaGlu7O_CFc0Wz74HITzWs4UEa4?usp=sharing">Colab Demo</a>
</div>

<p align="center">
  <a href="https://docs.nomic.ai">:closed_book:	 Atlas Python Client Documentation</a> 
  <br> <br>
      <a href="https://discord.gg/myY5YDR8z8">:hut: Discord</a> 
  <br> <br>
  <b>Example Maps</b>
  <br> <br>
  <a href="https://atlas.nomic.ai/map/twitter">:world_map: Map of Twitter</a> (5.4 million tweets)
  <br> <br>
  <a href="https://atlas.nomic.ai/map/stablediffusion">:world_map: Map of StableDiffusion Generations</a> (6.4 million images)
  <br> <br>
  <a href="https://atlas.nomic.ai/map/neurips">:world_map: Map of NeurIPS Proceedings</a> (16,623 abstracts)

</p>

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
