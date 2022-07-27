Nomic Client
========================================
.. toctree::
   :hidden:
   :maxdepth: 2

   modules


Quickstart
--------------------------

Install the Nomic client with:

.. code-block:: bash

   pip install nomic

Login/create your Nomic account:

.. code-block:: bash

   nomic login

Follow the instructions to obtain your access token. Enter your access token with:

.. code-block:: bash

   nomic login [token]

You are ready to interact with Nomic services.

Your first neural map
########

The following code snippet shows you how to map your embeddings with Atlas - Nomic's neural database.
Upload 10,000 random embeddings and see them instantly organized on an interactive map.

.. literalinclude:: ../../examples/map_embeddings.py
    :language: python


Your first neural map of text
########

Neural maps let you see data through the eyes of a neural network. First we need access to a powerful neural network.
Cohere AI provides an embedding API to a powerful large language model to get you started.
Obtain an API key from `cohere.ai <https://os.cohere.ai/>`_ to embed your text data.

Add your Cohere API key to the below example to see how their large language model organizes text from a sentiment analysis dataset.

.. literalinclude:: ../../examples/map_hf_dataset_with_cohere.py
    :language: python



