
Map your data
--------------------------
Nomics neural database ingests unstructured data such as embeddings or text and organizes them.
Once your data is in Atlas, you can view *all* of it at once on a neural map.

Your first neural map
#####################

The following code snippet shows you how to map your embeddings with Atlas - Nomic's neural database.
Upload 10,000 random embeddings and see them instantly organized on an interactive map.

.. literalinclude:: ../../examples/map_embeddings.py
    :language: python

This code will return a link to view your neural map.

Your first neural map of text
#############################

Neural maps let you see data through the eyes of a neural network. First we need access to a powerful neural network.
Cohere AI's large language model embedding API will get you started.
Obtain an API key from `cohere.ai <https://os.cohere.ai/>`_ to embed your text data.

Add your Cohere API key to the below example to see how their large language model organizes text from a sentiment analysis dataset.

.. literalinclude:: ../../examples/map_hf_dataset_with_cohere.py
    :language: python
