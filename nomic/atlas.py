"""
This class allows for programmatic interactions with Atlas - Nomic's neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""
import concurrent.futures
import json
from typing import Dict, List, Optional

import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from wonderwords import RandomWord

from .cli import get_api_credentials

# Uploads send several requests to allow for threadpool refreshing.
# Threadpool hogs memory and new ones need to be created.
# This number specifies how much data gets processed before a new Threadpool is created
MAX_MEMORY_CHUNK = 150000


class CreateIndexResponse(BaseModel):
    map: Optional[str] = Field(
        None, description="A link to the map this index creates. May take some time to be ready so check the job state."
    )
    job_id: str = Field(..., description="The job_id to track the progress of the index build.")
    index_id: str = Field(..., description="The unique identifier of the index being built.")
    project_id: str = Field(..., description="The id of the project this map is being created in")


class AtlasClient:
    """The Atlas Client"""

    def __init__(self):
        '''
        Initializes the Atlas client.

        '''

        self.credentials = get_api_credentials()

        if self.credentials['tenant'] == 'staging':
            hostname = 'staging-api-atlas.nomic.ai'
        elif self.credentials['tenant'] == 'production':
            hostname = 'api-atlas.nomic.ai'
        else:
            raise ValueError("Invalid tenant.")

        self.atlas_api_path = f"https://{hostname}"
        self.token = self.credentials['token']
        self.header = {"Authorization": f"Bearer {self.token}"}

        if self.token:
            response = requests.get(
                self.atlas_api_path + "/v1/user",
                headers=self.header,
            )
            if not response.status_code == 200:
                logger.info("Your authorization token is no longer valid.")
        else:
            raise ValueError(
                "Could not find an authorization token. Run `nomic login` to authorize this client with the Nomic API."
            )

    def _get_current_user(self):
        response = requests.get(
            self.atlas_api_path + "/v1/user",
            headers=self.header,
        )
        if not response.status_code == 200:
            raise ValueError("Your authorization token is no longer valid. Run `nomic login` to obtain a new one.")

        return response.json()

    def create_project(
        self,
        project_name: str,
        description: str,
        unique_id_field: str,
        modality: str,
        organization_name: str = None,
        is_public: bool = True,
    ):
        '''
        Creates an Atlas project. Atlas projects store data (text, embeddings, etc) that you can organize by building indices.

        **Parameters:**

        * **project_name** - The name of the project.
        * **description** - A description for the project.
        * **unique_id_field** - The field that uniquely identifies each datum. If a datum does not contain this field, it will be added and assigned a random unique ID.
        * **modality** - The data modality of this project. Currently, Atlas supports either `text` or `embedding` modality projects.
        * **organization_name** - The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
        * **is_public** - Should this project be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.

        **Returns:** project_id on success.

        '''
        supported_modalities = ['text', 'embedding']
        if modality not in supported_modalities:
            msg = 'Tried to create project with modality: {}, but Atlas only supports: {}'.format(
                modality, supported_modalities
            )
            raise ValueError(msg)

        if organization_name is None:
            organization = self._get_current_users_main_organization()
            organization_name = organization['nickname']
            organization_id = organization['organization_id']
        else:
            organization_id_request = requests.get(
                self.atlas_api_path + f"/v1/organization/search/{organization_name}", headers=self.header
            )
            if organization_id_request.status_code != 200:
                user = self._get_current_user()
                users_organizations = [org['nickname'] for org in user['organizations']]
                raise Exception(
                    f"No such organization exists: {organization_name}. You can add projects to the following organization: {users_organizations}"
                )
            organization_id = organization_id_request.json()['organization_id']

        logger.info(f"Creating project `{project_name}` in organization `{organization_name}`")

        response = requests.post(
            self.atlas_api_path + "/v1/project/create",
            headers=self.header,
            json={
                'organization_id': organization_id,
                'project_name': project_name,
                'description': description,
                'unique_id_field': unique_id_field,
                'modality': modality,
                'is_public': is_public,
            },
        )
        if response.status_code != 201:
            raise Exception(f"Failed to create project: {response.json()}")
        return response.json()['project_id']

    def _get_current_users_main_organization(self):
        '''
        Retrieves the ID of the current users default organization.

        **Returns:** The ID of the current users default organization

        '''

        user = self._get_current_user()
        for organization in user['organizations']:
            if organization['user_id'] == user['sub'] and organization['access_role'] == 'OWNER':
                return organization

    def _get_project_by_name(self, project_name: str):
        '''
        Retrieves a project that a user owns by name.

        **Parameters:**

        * **project_name** - The name of the project.

        **Returns:** the project.

        Returns:

        '''

        organization_id = self._get_current_users_main_organization()['organization_id']

        organization = requests.get(
            self.atlas_api_path + f"/v1/organization/{organization_id}",
            headers=self.header,
        ).json()

        target_project = None
        for candidate in organization['projects']:
            if candidate['project_name'] == project_name:
                target_project = candidate
                break

        if target_project is None:
            raise ValueError(f"Could not find project `{project_name}`")

        return target_project

    def _get_project_by_id(self, project_id: str):
        '''

        Args:
            project_id: The project id

        Returns:
            Returns the requested project.
        '''

        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project_id}",
            headers=self.header,
        )

        if response.status_code != 200:
            raise Exception(f"Could not access project: {response.json()}")

        return response.json()

    def _get_index_job(self, job_id: str):
        '''

        Args:
            job_id: The job id to retrieve the state of.

        Returns:
            Job ID meta-data.
        '''

        response = requests.get(
            self.atlas_api_path + f"/v1/project/index/job/{job_id}",
            headers=self.header,
        )

        if response.status_code != 200:
            raise Exception(f'Could not access job state: {response.json()}')

        return response.json()

    def add_embeddings(
        self, project_id: str, embeddings: np.array, data: List[Dict], shard_size=1000, num_workers=10, pbar=None
    ):
        '''
        Adds embeddings to an embedding project. Pair each embedding with meta-data to explore your embeddings.

        **Parameters:**

        * **project_id** - The id of the project you are adding embeddings to.
        * **embeddings** - An [N,d] numpy array containing the batch of N embeddings to add.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.

        **Returns:** True on success.
        '''

        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        # Check if this is a progressive project
        response = requests.get(
            self.atlas_api_path+ f"/v1/project/{project_id}",
            headers=self.header,
        )
        progressive = len(response.json()['atlas_indices']) > 0
        upload_endpoint = "/v1/project/data/add/embedding/initial"
        if progressive:
            upload_endpoint = "/v1/project/data/add/embedding/progressive"

        # Actually do the upload
        def send_request(i):
            data_shard = data[i : i + shard_size]
            embedding_shard = embeddings[i : i + shard_size, :].tolist()
            response = requests.post(
                self.atlas_api_path + upload_endpoint,
                headers=self.header,
                json={'project_id': project_id, 'embeddings': embedding_shard, 'data': data_shard},
            )
            del embedding_shard
            return response

        failed = []

        # if this method is being called internally, we pass a global progress bar
        close_pbar = False
        if pbar is None:
            logger.info("Uploading embeddings to Nomic.")
            close_pbar = True
            pbar = tqdm(total=int(embeddings.shape[0]) // shard_size)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(send_request, i): i for i in range(0, len(data), shard_size)}
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                pbar.update(1)
                if response.status_code != 200:
                    logger.error(f"Shard upload failed: {response.json()}")
                    if 'more datums exceeds your organization limit' in response.json():
                        return False

                    failed.append(futures[future])
        # close the progress bar if this method was called with no external progresbar
        if close_pbar:
            pbar.close()

        if failed:
            logger.warning(f"Failed to upload {len(failed)*shard_size} datums")
        if close_pbar:
            if failed:
                logger.warning("Embedding upload partially succeeded.")
            else:
                logger.warning("Embedding upload succeeded.")

        return True

    def create_index(self, project_id: str, index_name: str, indexed_field=None, colorable_fields=[]):
        '''
        Creates an index in the specified project

        **Parameters:**

        * **project_id** - The ID of the project this index is being built under.
        * **index_name** - The name of the index
        * **indexed_field** - Default None. For text projects, name the data field corresponding to the text to be mapped.
        * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.

        **Returns:** A link to your map.
        '''

        project = self._get_project_by_id(project_id=project_id)

        if project['modality'] == 'embedding':
            embedding_build_template = {
                'project_id': project['id'],
                'index_name': index_name,
                'indexed_field': None,
                'atomizer_strategies': None,
                'model': None,
                'colorable_fields': colorable_fields,
                'model_hyperparameters': None,
                'nearest_neighbor_index': 'HNSWIndex',
                'nearest_neighbor_index_hyperparameters': json.dumps({'space': 'l2', 'ef_construction': 100, 'M': 16}),
                'projection': 'NomicProject',
                'projection_hyperparameters': json.dumps(
                    {'n_neighbors': 15, 'min_dist': 3e-2, 'force_approximation_algorithm': True}
                ),
            }

            response = requests.post(
                self.atlas_api_path + "/v1/project/index/create",
                headers=self.header,
                json=embedding_build_template,
            )
            job_id = response.json()['job_id']

        elif project['modality'] == 'text':
            hyperparameters = {
                'dataset_buffer_size': 1000,
                'batch_size': 4,
                'polymerize_by': 'charchunk',
            }
            text_build_template = {
                'project_id': project['id'],
                'index_name': index_name,
                'indexed_field': indexed_field,
                'atomizer_strategies': ['document', 'charchunk'],
                'model': 'NomicEmbed',
                'colorable_fields': colorable_fields,
                'model_hyperparameters': json.dumps(hyperparameters),
                'nearest_neighbor_index': 'HNSWIndex',
                'nearest_neighbor_index_hyperparameters': json.dumps({'space': 'l2', 'ef_construction': 100, 'M': 16}),
                'projection': 'NomicProject',
                'projection_hyperparameters': json.dumps(
                    {'n_neighbors': 15, 'min_dist': 3e-2, 'force_approximation_algorithm': True}
                ),
            }
            response = requests.post(
                self.atlas_api_path + "/v1/project/index/create",
                headers=self.header,
                json=text_build_template,
            )
            job_id = response.json()['job_id']

        job = requests.get(
            self.atlas_api_path + f"/v1/project/index/job/{job_id}",
            headers=self.header,
        ).json()

        index_id = job['index_id']

        project = requests.get(
            self.atlas_api_path + f"/v1/project/{project['id']}",
            headers=self.header,
        ).json()

        projection_id = None

        for index in project['atlas_indices']:
            if index['id'] == index_id:
                projection_id = index['projections'][0]['id']
                break

        to_return = {'job_id': job_id, 'index_id': index_id}
        if not projection_id:
            logger.warning("Could not find a projection being built for this index.")
        else:
            if self.credentials['tenant'] == 'staging':
                to_return['map'] = f"https://staging-atlas.nomic.ai/map/{project['id']}/{projection_id}"
            else:
                to_return['map'] = f"https://atlas.nomic.ai/map/{project['id']}/{projection_id}"
            logger.info(f"Created map `{index_name}`: {to_return['map']}")
        to_return['project_id'] = project['id']
        return CreateIndexResponse(**to_return)

    def add_text(self, project_id: str, data: List[Dict], shard_size=1000, num_workers=10, pbar=None):
        '''
        Adds data to a text project.

        **Parameters:**

        * **project_id** - The id of the project you are adding embeddings to.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.

        **Returns:** True on success.
        '''

        # Ensure there are no empty datums
        for i, elem in enumerate(data):
            for k, v in elem.items():
                if isinstance(v, str) and len(v) == 0:
                    msg = 'Datum number: {} had an empty string for key: {}'.format(i, k)
                    raise ValueError(msg)

        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        # Check if this is a progressive project
        response = requests.get(
            self.atlas_api_path+ f"/v1/project/{project_id}",
            headers=self.header,
        )
        progressive = len(response.json()['atlas_indices']) > 0
        upload_endpoint = "/v1/project/data/add/json/initial"
        if progressive:
            upload_endpoint = "/v1/project/data/add/json/progressive"

        # Actually do the upload
        def send_request(i):
            data_shard = data[i : i + shard_size]
            response = requests.post(
                self.atlas_api_path + upload_endpoint,
                headers=self.header,
                json={'project_id': project_id, 'data': data_shard},
            )
            return response

        failed = []

        # if this method is being called internally, we pass a global progress bar
        close_pbar = False
        if pbar is None:
            logger.info("Uploading text to Nomic.")
            close_pbar = True
            pbar = tqdm(total=int(len(data)) // shard_size)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(send_request, i): i for i in range(0, len(data), shard_size)}
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                pbar.update(1)
                if response.status_code != 200:
                    try:
                        logger.error(f"Shard upload failed: {response.json()}")
                        if 'more datums exceeds your organization limit' in response.json():
                            return False
                    except requests.exceptions.JSONDecodeError:
                        logger.error(f"Shard upload failed: {response}")
                        continue

                    failed.append(futures[future])
        # close the progress bar if this method was called with no external progresbar
        if close_pbar:
            pbar.close()

        if failed:
            logger.warning(f"Failed to upload {len(failed)*shard_size} datums")
        if close_pbar:
            if failed:
                logger.warning("Text upload partially succeeded.")
            else:
                logger.warning("Text upload succeeded.")

        return True

    def map_embeddings(
        self,
        embeddings: np.array,
        data: List[Dict],
        id_field: str = 'id',
        is_public: bool = True,
        colorable_fields: list = [],
        num_workers: int = 10,
        map_name: str = None,
        map_description: str = None,
        organization_name: str = None,
    ):
        '''
        Generates a map of the given embeddings.

        **Parameters:**

        * **embeddings** - An [N,d] numpy array containing the batch of N embeddings to add.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **id_field** - Each datums unique id field.
        * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        * **is_public** - Should this embedding map be public? Private maps can only be accessed by members of your organization.
        * **num_workers** - The number of workers to use when sending data.
        * **map_name** - A name for your map.
        * **map_description** - A description for your map.
        * **organization_name** - *(optional)* The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.

        **Returns:** A link to your map.
        '''

        def get_random_name():
            random_words = RandomWord()
            return f"{random_words.word(include_parts_of_speech=['adjectives'])}-{random_words.word(include_parts_of_speech=['nouns'])}"

        project_name = get_random_name()
        description = project_name
        index_name = get_random_name()

        if map_name:
            project_name = map_name
        if map_description:
            description = map_description

        if id_field in colorable_fields:
            raise Exception(f'Cannot color by unique id field: {id_field}')
        for field in colorable_fields:
            if field not in data[0]:
                raise Exception(f"Cannot color by field `{field}` as it is not present in the meta-data.")

        project_id = self.create_project(
            project_name=project_name,
            description=description,
            unique_id_field=id_field,
            modality='embedding',
            is_public=is_public,
            organization_name=organization_name,
        )

        shard_size = 1000
        if embeddings.shape[0] > 10000:
            shard_size = 2500

        # sends several requests to allow for threadpool refreshing. Threadpool hogs memory and new ones need to be created.
        logger.info("Uploading embeddings to Nomic's neural database Atlas.")

        embeddings = embeddings.astype(np.float16)

        with tqdm(total=len(data) // shard_size) as pbar:
            for i in range(0, len(data), MAX_MEMORY_CHUNK):
                self.add_embeddings(
                    project_id=project_id,
                    embeddings=embeddings[i : i + MAX_MEMORY_CHUNK, :],
                    data=data[i : i + MAX_MEMORY_CHUNK],
                    shard_size=shard_size,
                    num_workers=num_workers,
                    pbar=pbar,
                )

        logger.info("Embedding upload succeeded.")

        response = self.create_index(project_id=project_id, index_name=index_name, colorable_fields=colorable_fields)

        return dict(response)

    def update_maps(self,
                    project_id: str,
                    data: List[Dict],
                    embeddings: Optional[np.array]=None,
                    shard_size: int=1000,
                    num_workers: int = 10):
        '''
        Updates a project's maps with new data.

        **Parameters:**

        * **project_id** - The id of the project you want to update
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **embedding** - (Default None) An [N, d] matrix of embeddings for updating embedding projects. Leave as None to update text projects.
        * **shard_size** - Data is uploaded in parallel by many threads. Adjust the number of datums to upload by each worker.
        * **num_workers** - The number of workers to use when sending data.

        **Returns:** job_ids: The ids of the update jobs
        '''

        # Validate data
        project = self._get_project_by_id(project_id=project_id)
        if project['modality'] == 'embedding' and embeddings is None:
            msg = 'Please specify embeddings for updating an embedding project'
            raise ValueError(msg)

        if project['modality'] == 'text' and embeddings is not None:
            msg = 'Please dont specify embeddings for updating a text project'
            raise ValueError(msg)

        if embeddings is not None and len(data) != embeddings.shape[0]:
            msg = 'Expected data and embeddings to be the same length but found lengths {} and {} respectively.'.format()
            raise ValueError(msg)


        # Add new data
        logger.info("Uploading data to Nomic's neural database Atlas.")
        with tqdm(total=len(data) // shard_size) as pbar:
            for i in range(0, len(data), MAX_MEMORY_CHUNK):
                if project['modality'] == 'embedding':
                    self.add_embeddings(
                        project_id=project_id,
                        embeddings=embeddings[i: i + MAX_MEMORY_CHUNK, :],
                        data=data[i: i + MAX_MEMORY_CHUNK],
                        shard_size=shard_size,
                        num_workers=num_workers,
                        pbar=pbar,
                    )
                else:
                    self.add_text(
                        project_id=project_id,
                        data=data[i: i + MAX_MEMORY_CHUNK],
                        shard_size=shard_size,
                        num_workers=num_workers,
                        pbar=pbar,
                    )
        logger.info("Upload succeeded.")

        #Update maps
        # finally, update all the indices
        response = requests.post(
            self.atlas_api_path + "/v1/project/update_indices",
            headers=self.header,
            json={
                'project_id': str(project_id),
            },
        )

        return response.json()['job_ids']



    def map_text(
        self,
        data: List[Dict],
        indexed_field: str,
        id_field: str = 'id',
        is_public: bool = True,
        colorable_fields: list = [],
        num_workers: int = 10,
        map_name: str = None,
        map_description: str = None,
        organization_name: str = None,
    ):
        '''
        Generates a map of the given text.

        **Parameters:**

        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **indexed_field** - The name the data field corresponding to the text to be mapped.
        * **id_field** - Each datums unique id field.
        * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        * **is_public** - Should this embedding map be public? Private maps can only be accessed by members of your organization.
        * **num_workers** - The number of workers to use when sending data.
        * **map_name** - A name for your map.
        * **map_description** - A description for your map.
        * **organization_name** - *(optional)* The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.

        **Returns:** A link to your map.
        '''

        def get_random_name():
            random_words = RandomWord()
            return f"{random_words.word(include_parts_of_speech=['adjectives'])}-{random_words.word(include_parts_of_speech=['nouns'])}"

        project_name = get_random_name()
        description = project_name
        index_name = get_random_name()

        if map_name:
            project_name = map_name
        if map_description:
            description = map_description

        if id_field in colorable_fields:
            raise Exception(f'Cannot color by unique id field: {id_field}')

        project_id = self.create_project(
            project_name=project_name,
            description=description,
            unique_id_field=id_field,
            modality='text',
            is_public=is_public,
            organization_name=organization_name,
        )

        shard_size = 1000
        if len(data) > 10000:
            shard_size = 2500

        logger.info("Uploading text to Nomic's neural database Atlas.")

        with tqdm(total=len(data) // shard_size) as pbar:
            for i in range(0, len(data), MAX_MEMORY_CHUNK):
                self.add_text(
                    project_id=project_id,
                    data=data[i : i + MAX_MEMORY_CHUNK],
                    shard_size=shard_size,
                    num_workers=num_workers,
                    pbar=pbar,
                )

        logger.info("Text upload succeeded.")

        response = self.create_index(
            project_id=project_id, indexed_field=indexed_field, index_name=index_name, colorable_fields=colorable_fields
        )

        return dict(response)
