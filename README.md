<h1 align="center">Nomic</h1>
<p align="center">The Python client to Atlas - the world's first neural database.</p>

<p align="center">
  <a href="https://docs.nomic.ai">Documentation</a> 
  <br> <br>
  <br> <br>
</p>

[//]: # (<img src="" alt="Nomic Workflow" style="display: block; margin: 0 auto;" />)

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

Explore the [documentation](https://docs.nomic.ai) to make more advanced maps.
