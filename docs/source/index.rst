Nomic Client Documentation
========================================
.. toctree::
   :hidden:
   :maxdepth: 2

   modules

Quickstart
--------------------------
Learn how to login and interact with Nomic services.

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

The following code snippet provides an example interaction with Atlas - Nomic's neural database.
Upload 10,000 embeddings and see them instantly organized on an interactive map.

.. literalinclude:: ../../examples/map_embeddings.py
    :language: python

