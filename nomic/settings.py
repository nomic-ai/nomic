# Uploads send several requests to allow for threadpool refreshing.
# Threadpool hogs memory and new ones need to be created.
# This number specifies how much data gets processed before a new Threadpool is created
MAX_MEMORY_CHUNK = 150000
EMBEDDING_PAGINATION_LIMIT = 1000
IMAGE_EMBEDDING_BATCH_SIZE = 64
ATLAS_DEFAULT_ID_FIELD = 'id_'

DEFAULT_PROJECTION_MODEL = 'nomic-project-v1'

DEFAULT_DUPLICATE_THRESHOLD = 0.1
