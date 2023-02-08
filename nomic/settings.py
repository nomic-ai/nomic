# Uploads send several requests to allow for threadpool refreshing.
# Threadpool hogs memory and new ones need to be created.
# This number specifies how much data gets processed before a new Threadpool is created
MAX_MEMORY_CHUNK = 150000
EMBEDDING_PAGINATION_LIMIT = 1000
ATLAS_DEFAULT_ID_FIELD = 'id_'

DEFAULT_PROJECTION_N_NEIGHBORS = 15
DEFAULT_PROJECTION_EPOCHS = 50
DEFAULT_LARGE_PROJECTION_N_NEIGHBORS = 128
DEFAULT_LARGE_PROJECTION_EPOCHS = 128

DEFAULT_PROJECTION_SPREAD = 1.0
