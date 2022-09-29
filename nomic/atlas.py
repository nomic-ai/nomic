"""
This class allows for programmatic interactions with Atlas - Nomic's neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""
import os
import pickle
import concurrent.futures
import json
import uuid
import gc
from typing import Dict, List, Optional

import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from wonderwords import RandomWord

from .cli import get_api_credentials, refresh_bearer_token, validate_api_http_response
import sys
# Uploads send several requests to allow for threadpool refreshing.
# Threadpool hogs memory and new ones need to be created.
# This number specifies how much data gets processed before a new Threadpool is created
MAX_MEMORY_CHUNK = 150000
EMBEDDING_PAGINATION_LIMIT = 1000

def get_object_size_in_bytes(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

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

        refresh_bearer_token()
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
            response = validate_api_http_response(response)
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
        response = validate_api_http_response(response)
        if not response.status_code == 200:
            raise ValueError("Your authorization token is no longer valid. Run `nomic login` to obtain a new one.")

        return response.json()

    def _ensure_metadata(self, metadata):
        keys = metadata[0].keys()
        for key in keys:
            if len(key) >=2 and key[:2] == '__':
                raise ValueError('Metadata fields cannot start with __')

        keylist = sorted(list(keys))
        for datum in metadata:
            cur_keylist = sorted(list(datum.keys()))
            if cur_keylist != keylist:
                msg = 'All metadata must have the same keys, but found key sets: {} and {}'.format(keylist, cur_keylist)
                raise ValueError(msg)

        return True

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

    def get_project_by_id(self, project_id: str):
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

    def _validate_user_supplied_metadata(self, data: List[Dict], project, replace_empty_string_values_with_string_null=True):
        '''
        Validates the users metadata for Atlas compatability.
        If unique_id_field is specified, validates that each datum has that field. If not, adds it
        and then notifies the user that it was added.
        Args:
            data: the user supplied list of data dictionaries
            project: the atlas project you are validating the data for.
            replace_empty_string_values_with_string_null: replaces empty string values with string nulls in all datums

        Returns:

        '''

        if not isinstance(data, list):
            raise Exception("Metadata must be a list of dictionaries")

        unique_id_field = project['unique_id_field']
        metadata_keys = None
        added_id_field_for_user = False
        for datum in data:
            if not isinstance(datum, dict):
                raise Exception('Each metadata must be a dictionary with one level of keys and values of only string, int and float types.')

            if unique_id_field not in datum:
                added_id_field_for_user = True
                datum[unique_id_field] = str(uuid.uuid4())

            if metadata_keys is None:
                metadata_keys = sorted(list(datum.keys()))

            datum_keylist = sorted(list(datum.keys()))
            if datum_keylist != metadata_keys:
                msg = 'All metadata must have the same keys, but found key sets: {} and {}'.format(metadata_keys, datum_keylist)
                raise ValueError(msg)

            for key in datum:
                if key.startswith('__'):
                    raise ValueError('Metadata fields cannot start with __')

                if project['modality'] == 'text':
                    if isinstance(datum[key], str) and len(datum[key]) == 0:
                        if replace_empty_string_values_with_string_null:
                            datum[key] = 'null'
                        else:
                            msg = 'Datum {} had an empty string for key: {}'.format(datum, key)
                            raise ValueError(msg)

                if not isinstance(datum[key], (str, float, int)):
                    raise Exception(f"Metadata sent to Atlas must be a flat dictionary. Values must be strings, floats or ints. Key `{key}` of datum {str(datum)} is in violation.")

        if added_id_field_for_user:
            logger.info(f"A datum you supplied lacked the unique id field `{unique_id_field}`. Added it for you.")


    def is_project_accepting_data(self, project_id: str):
        '''
        Checks if the project can accept data. Projects cannot accept data when they are being indexed.

        **Parameters:**

        * **project_id** - The id of the project you are checking.

        **Returns:** True if project is unlocked for data additions, false otherwise.
        '''
        response = requests.get(
            self.atlas_api_path+ f"/v1/project/{project_id}",
            headers=self.header,
        )
        project = response.json()

        return not project['insert_update_delete_lock']


    def add_embeddings(
        self, project_id: str, embeddings: np.array, data: List[Dict], shard_size=1000, num_workers=10, replace_empty_string_values_with_string_null=True, pbar=None
    ):
        '''
        Adds embeddings to an embedding project. Pair each embedding with meta-data to explore your embeddings.

        **Parameters:**

        * **project_id** - The id of the project you are adding embeddings to.
        * **embeddings** - An [N,d] numpy array containing the batch of N embeddings to add.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.
        * **replace_empty_string_values_with_string_null** - Replaces empty values in metadata with null. If false, will fail if empty values are supplied.

        **Returns:** True on success.
        '''

        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        # Check if this is a progressive project
        response = requests.get(
            self.atlas_api_path+ f"/v1/project/{project_id}",
            headers=self.header,
        )
        project = response.json()

        if project['modality'] != 'embedding':
            msg = 'Cannot add embedding to project with modality: {}'.format(project['modality'])
            raise ValueError(msg)

        if project['insert_update_delete_lock']:
            raise Exception("Project is currently indexing and cannot ingest new datums. Try again later.")

        progressive = len(project['atlas_indices']) > 0
        try:
            self._validate_user_supplied_metadata(data=data, project=project, replace_empty_string_values_with_string_null=replace_empty_string_values_with_string_null)
        except BaseException as e:
            raise e


        upload_endpoint = "/v1/project/data/add/embedding/initial"
        if progressive:
            upload_endpoint = "/v1/project/data/add/embedding/progressive"

        # Actually do the upload
        def send_request(i):
            data_shard = data[i : i + shard_size]

            if get_object_size_in_bytes(data_shard) > 8000000:
                raise Exception("Your metadata upload shards are to large. Try decreasing the shard size or removing un-needed fields from the metadata.")
            self._ensure_metadata(data_shard)
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
                    try:
                        logger.error(f"Shard upload failed: {response.json()}")
                        if 'more datums exceeds your organization limit' in response.json():
                            return False
                        if 'Project transaction lock is held':
                            raise Exception("Project is currently indexing and cannot ingest new datums. Try again later.")
                    except requests.exceptions.JSONDecodeError:
                        if response.status_code == 413:
                            logger.error("Shard upload failed: you are sending meta-data data is to large.")
                        else:
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

        project = self.get_project_by_id(project_id=project_id)

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
            if indexed_field is None:
                raise Exception("You did not specify a field to index. Specify an 'indexed_field'.")

            if indexed_field not in project['project_fields']:
                raise Exception(f"Your index field is not valid. Valid options are: {project['project_fields']}")

            hyperparameters = {
                'dataset_buffer_size': 1000,
                'batch_size': 20,
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
                    {'n_neighbors': 15, 'n_epochs': 50, 'spread': 1}
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
            logger.info(f"Created map `{index_name}` in project `{project['project_name']}`: {to_return['map']}")
        to_return['project_id'] = project['id']
        return CreateIndexResponse(**to_return)

    def add_text(self, project_id: str, data: List[Dict], shard_size=1000, num_workers=10, replace_empty_string_values_with_string_null=True, pbar=None):
        '''
        Adds data to a text project.

        **Parameters:**

        * **project_id** - The id of the project you are adding embeddings to.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.
        * **replace_empty_string_values_with_string_null** - Replaces empty values in metadata with null. If false, will fail if empty values are supplied.

        **Returns:** True on success.
        '''


        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        # Check if this is a progressive project
        response = requests.get(
            self.atlas_api_path+ f"/v1/project/{project_id}",
            headers=self.header,
        )

        project = response.json()
        if project['modality'] != 'text':
            msg = 'Cannot add text to project with modality: {}'.format(project['modality'])
            raise ValueError(msg)

        progressive = len(project['atlas_indices']) > 0

        if project['insert_update_delete_lock']:
            raise Exception("Project is currently indexing and cannot ingest new datums. Try again later.")

        try:
            self._validate_user_supplied_metadata(data=data, project=project,
                                                  replace_empty_string_values_with_string_null=replace_empty_string_values_with_string_null)
        except BaseException as e:
            raise e

        upload_endpoint = "/v1/project/data/add/json/initial"
        if progressive:
            upload_endpoint = "/v1/project/data/add/json/progressive"

        # Actually do the upload
        def send_request(i):
            data_shard = data[i : i + shard_size]
            if get_object_size_in_bytes(data_shard) > 8000000:
                raise Exception("Your metadata upload shards are to large. Try decreasing the shard size or removing un-needed fields from the metadata.")
            self._ensure_metadata(data_shard)
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
                        if 'Project transaction lock is held':
                            raise Exception("Project is currently indexing and cannot ingest new datums. Try again later.")
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
                logger.info("Text upload succeeded.")

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
            index_name = map_name
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
                try:
                    self.add_embeddings(
                        project_id=project_id,
                        embeddings=embeddings[i : i + MAX_MEMORY_CHUNK, :],
                        data=data[i : i + MAX_MEMORY_CHUNK],
                        shard_size=shard_size,
                        num_workers=num_workers,
                        pbar=pbar,
                    )
                except BaseException as e:
                    self.delete_project(project_id=project_id)
                    raise e


        logger.info("Embedding upload succeeded.")

        response = self.create_index(project_id=project_id, index_name=index_name, colorable_fields=colorable_fields)

        return {**dict(response), 'project_id': project_id}

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
        project = self.get_project_by_id(project_id=project_id)
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
        return self.refresh_maps(project_id=project_id)

    def refresh_maps(self, project_id: str):
        '''
        Refresh all maps in a project with the latest state.

        **Parameters:**

        * **project_id** - The id of the project whose maps will be refreshed.

        **Returns:** a list of jobs
        '''

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
            index_name = map_name
        if map_description:
            description = map_description

        if id_field in colorable_fields:
            raise Exception(f'Cannot color by unique id field: {id_field}')
        if id_field not in data[0]:
            raise Exception(f"You specified `{id_field}` as your unique id field but it is not contained in your data upload")

        project_id = self.create_project(
            project_name=project_name,
            description=description,
            unique_id_field=id_field,
            modality='text',
            is_public=is_public,
            organization_name=organization_name,
        )

        shard_size = 1000

        logger.info("Uploading text to Nomic's neural database Atlas.")

        with tqdm(total=len(data) // shard_size) as pbar:
            for i in range(0, len(data), MAX_MEMORY_CHUNK):
                try:
                    self.add_text(
                        project_id=project_id,
                        data=data[i : i + MAX_MEMORY_CHUNK],
                        shard_size=shard_size,
                        num_workers=num_workers,
                        pbar=pbar,
                    )
                except BaseException as e:
                    self.delete_project(project_id=project_id)
                    raise e

        logger.info("Text upload succeeded.")

        response = self.create_index(
            project_id=project_id, indexed_field=indexed_field, index_name=index_name, colorable_fields=colorable_fields
        )

        return dict(response)


    def delete_project(self, project_id: str):
        '''
        Deletes an atlas project with all associated metadata.

        **Parameters:**

        * **project_id** - The id of the project you want to delete.
        '''
        organization = self._get_current_users_main_organization()
        organization_name = organization['nickname']

        project = self.get_project_by_id(project_id=project_id)


        logger.info(f"Deleting project `{project['project_name']}` from organization `{organization_name}`")

        response = requests.post(
            self.atlas_api_path + "/v1/project/remove",
            headers=self.header,
            json={'project_id': project_id},
        )

    def download_embeddings(self, project_id, atlas_index_id, output_dir, num_workers=10):
        '''
        Downloads a mapping from datum_id to embedding in shards to the provided directory

        Args:
            project_id: the id of the relevant index's parent project
            atlas_index_id: the id of the index whose ambient embeddings you want
            output_dir: the directory to save shards to


        '''

        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project_id}",
            headers=self.header,
        )
        project = response.json()
        print('project: ', project)
        total_datums = project['total_datums_in_project']
        if project['insert_update_delete_lock']:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT

        def download_shard(offset):
            response = requests.get(
                self.atlas_api_path + f"/v1/project/data/get/embedding/{project_id}/{atlas_index_id}/{offset}/{limit}",
                headers=self.header,
            )
            try:
                content = response.json()
                shard_name = '{}_{}_{}.pkl'.format(atlas_index_id, offset, offset+limit)
                shard_path = os.path.join(output_dir, shard_name)
                with open(shard_path, 'wb') as f:
                    pickle.dump(content, f)

            except Exception as e:
                logger.error('Shard {} download failed with error: {}'.format(shard_name, e))


        with tqdm(total=total_datums // limit) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(download_shard, cur_offset): cur_offset for cur_offset in range(0, total_datums, limit)}
                for future in concurrent.futures.as_completed(futures):
                    _ = future.result()
                    pbar.update(1)

    def get_embedding_iterator(self, project_id, atlas_index_id):
        '''
        Returns an iterable of datum_ids and embeddings from the given index

        Args:
            project_id: the id of the relevant index's parent project
            atlas_index_id: the id of the index whose ambient embeddings you want

        Returns:
            Iterable[Tuple[datum_ids, embeddings]]
        '''

        response = requests.get(
            self.atlas_api_path+ f"/v1/project/{project_id}",
            headers=self.header,
        )
        project = response.json()
        print('project: ', project)
        if project['insert_update_delete_lock']:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT
        while True:
            response = requests.get(
                self.atlas_api_path+ f"/v1/project/data/get/embedding/{project_id}/{atlas_index_id}/{offset}/{limit}",
                headers=self.header,
            )

            print('response: ', response)
            content = response.json()
            if len(content['datum_ids']) == 0:
                break
            offset += len(content['datum_ids'])

            yield content['datum_ids'], content['embeddings']