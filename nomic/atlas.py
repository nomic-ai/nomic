"""
This class allows for programmatic interactions with Atlas - Nomic's neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""
import os
import pickle
import concurrent.futures
import json
import uuid
import io
import time
import base64
from typing import Dict, List, Optional
from uuid import UUID

import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from datetime import date

from .utils import get_random_name, assert_valid_project_id, get_object_size_in_bytes
from .cli import get_api_credentials, refresh_bearer_token, validate_api_http_response

import sys

# Uploads send several requests to allow for threadpool refreshing.
# Threadpool hogs memory and new ones need to be created.
# This number specifies how much data gets processed before a new Threadpool is created
MAX_MEMORY_CHUNK = 150000
EMBEDDING_PAGINATION_LIMIT = 1000
ATLAS_DEFAULT_ID_FIELD = 'id_'


DEFAULT_PROJECTION_N_NEIGHBORS = 15
DEFAULT_PROJECTION_EPOCHS = 50
DEFAULT_PROJECTION_SPREAD = 1.0


class CreateIndexResponse(BaseModel):
    map: Optional[str] = Field(
        None, description="A link to the map this index creates. May take some time to be ready so check the job state."
    )
    job_id: str = Field(..., description="The job_id to track the progress of the index build.")
    index_id: str = Field(..., description="The unique identifier of the index being built.")
    project_id: str = Field(..., description="The id of the project this map is being created in")
    project_name: str = Field(..., description="The name of the project that was created.")


class AtlasUser:
    def __init__(self):
        self.credentials = refresh_bearer_token()


class AtlasClass(object):
    def __init__(self):
        '''
        Initializes the Atlas client.
        '''

        if self.credentials['tenant'] == 'staging':
            hostname = 'staging-api-atlas.nomic.ai'
        elif self.credentials['tenant'] == 'production':
            hostname = 'api-atlas.nomic.ai'
        else:
            raise ValueError("Invalid tenant.")

        self.atlas_api_path = f"https://{hostname}"
        token = self.credentials['token']
        self.token = token

        self.header = {"Authorization": f"Bearer {token}"}

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

    @property
    def credentials(self):
        return refresh_bearer_token()

    def _get_current_user(self):
        response = requests.get(
            self.atlas_api_path + "/v1/user",
            headers=self.header,
        )
        response = validate_api_http_response(response)
        if not response.status_code == 200:
            raise ValueError("Your authorization token is no longer valid. Run `nomic login` to obtain a new one.")

        return response.json()

    def _validate_map_data_inputs(self, colorable_fields, id_field, data):
        '''Validates inputs to map data calls.'''

        if not isinstance(colorable_fields, list):
            raise ValueError("colorable_fields must be a list of fields")

        if id_field in colorable_fields:
            raise Exception(f'Cannot color by unique id field: {id_field}')

        for field in colorable_fields:
            if field not in data[0]:
                raise Exception(f"Cannot color by field `{field}` as it is not present in the meta-data.")

    def _get_current_users_main_organization(self):
        '''
        Retrieves the ID of the current users default organization.

        **Returns:** The ID of the current users default organization

        '''

        user = self._get_current_user()
        for organization in user['organizations']:
            if organization['user_id'] == user['sub'] and organization['access_role'] == 'OWNER':
                return organization

    def get_project(self, project_name, organization_name=None):
        '''
        Retrieves a project by its name and organization.

        **Parameters:**

        * **project_name** - The id of the project you are checking.
        * **organization_name** - (Optional) The organization this project belongs to. Defaults to your main organization.

        **Returns:** A dictionary with details about your queried project. Error if project could not be found.
        '''

        if organization_name is None:
            organization = self._get_current_users_main_organization()
            organization_name = organization['nickname']
            organization_id = organization['organization_id']

        # check if this project already exists.
        response = requests.post(
            self.atlas_api_path + "/v1/project/search/name",
            headers=self.header,
            json={'organization_name': organization_name, 'project_name': project_name},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to search for project: {response.json()}")
        search_results = response.json()['results']

        if search_results:
            existing_project = search_results[0]
            return existing_project
        else:
            raise Exception(f"Could not find project `{project_name} in organization `{organization_name}``")

    def _get_project_by_id(self, project_id: str):
        '''

        Args:
            project_id: The project id

        Returns:
            Returns the requested project.
        '''

        assert_valid_project_id(project_id)

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

    def _validate_and_correct_user_supplied_metadata(
        self, data: List[Dict], project, replace_empty_string_values_with_string_null=True
    ):
        '''
        Validates the users metadata for Atlas compatability.

        1. If unique_id_field is specified, validates that each datum has that field. If not, adds it and then notifies the user that it was added.

        2. If a key is detected to store values that match an ISO8601 timestamp string ,Atlas will assume you are working with timestamps. If any additional metadata
        has this key associated with a non-ISO8601 timestamp string the upload will fail.

        Args:
            data: the user supplied list of data dictionaries
            project: the atlas project you are validating the data for.
            replace_empty_string_values_with_string_null: replaces empty string values with string nulls in all datums

        Returns:

        '''
        if not isinstance(data, list):
            raise Exception("Metadata must be a list of dictionaries")

        metadata_keys = None
        metadata_date_keys = []

        for datum in data:
            # The Atlas client adds a unique datum id field for each user.
            # It does not overwrite the field if it exists, instead map creation fails.
            if project['unique_id_field'] in datum:
                if len(str(datum[project['unique_id_field']])) > 36:
                    raise ValueError(
                        f"{datum}\n The id_field `{datum[project['unique_id_field']]}` is greater than 36 characters. Atlas does not support id_fields longer than 36 characters."
                    )
            else:
                if project['unique_id_field'] == ATLAS_DEFAULT_ID_FIELD:
                    datum[project['unique_id_field']] = str(uuid.uuid4())
                else:
                    raise ValueError(
                        f"{datum} does not contain your specified id_field `{datum[project['unique_id_field']]}`"
                    )

            if not isinstance(datum, dict):
                raise Exception(
                    'Each metadata must be a dictionary with one level of keys and values of only string, int and float types.'
                )

            if metadata_keys is None:
                metadata_keys = sorted(list(datum.keys()))

                # figure out which are dates
                for key in metadata_keys:
                    try:
                        date.fromisoformat(str(datum[key]))
                        metadata_date_keys.append(key)
                    except ValueError:
                        pass

            datum_keylist = sorted(list(datum.keys()))
            if datum_keylist != metadata_keys:
                msg = 'All metadata must have the same keys, but found key sets: {} and {}'.format(
                    metadata_keys, datum_keylist
                )
                raise ValueError(msg)

            for key in datum:
                if key.startswith('_'):
                    raise ValueError('Metadata fields cannot start with _')

                if key in metadata_date_keys:
                    try:
                        date.fromisoformat(str(datum[key]))
                    except ValueError:
                        raise ValueError(
                            f"{datum} has timestamp key `{key}` which cannot be parsed as a ISO8601 string. See the following documentation in the Nomic client for working with timestamps: https://docs.nomic.ai/mapping_faq.html."
                        )

                if project['modality'] == 'text':
                    if isinstance(datum[key], str) and len(datum[key]) == 0:
                        if replace_empty_string_values_with_string_null:
                            datum[key] = 'null'
                        else:
                            msg = 'Datum {} had an empty string for key: {}'.format(datum, key)
                            raise ValueError(msg)

                if not isinstance(datum[key], (str, float, int)):
                    raise Exception(
                        f"Metadata sent to Atlas must be a flat dictionary. Values must be strings, floats or ints. Key `{key}` of datum {str(datum)} is in violation."
                    )

    def is_project_accepting_data(self, project_id: str):
        '''
        Checks if the project can accept data. Projects cannot accept data when they are being indexed.

        **Parameters:**

        * **project_id** - The id of the project you are checking.

        **Returns:** True if project is unlocked for data additions, false otherwise.
        '''
        assert_valid_project_id(project_id)

        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project_id}",
            headers=self.header,
        )
        project = response.json()

        return not project['insert_update_delete_lock']

    def get_tags(self, project_name: str, index_name=None):
        '''
        Retrieves back all tags made in the web browser for a specific project and map.

        **Parameters:**

        * **project_name** - The name of the project you are getting tags from.
        * **index_name** - The name of the atlas index in the project.

        **Returns:** A dictionary mapping datum ids to tags.
        '''

        project = self.get_project(project_name=project_name)

        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project['id']}",
            headers=self.header,
        )
        project = response.json()
        if len(project['atlas_indices']) == 0:
            raise Exception(f'There are no indices in the project `{project_name}`.')

        target_index = None
        for index in project['atlas_indices']:
            if index['index_name'] == index_name:
                target_index = index
                break
        if target_index is None:
            target_index = project['atlas_indices'][0]

        # now get the tags
        datums_and_tags = requests.post(
            self.atlas_api_path + '/v1/project/tag/read/all_by_datum',
            headers=self.header,
            json={
                'project_id': project['id'],
                'atlas_index_id': target_index['id'],
            },
        ).json()['results']

        datum_to_labels = {}
        for item in datums_and_tags:
            datum_to_labels[item['datum_id']] = item['labels']

        return datums_and_tags

    def map_embeddings(
        self,
        embeddings: np.array,
        data: List[Dict] = None,
        id_field: str = None,
        is_public: bool = True,
        colorable_fields: list = [],
        num_workers: int = 10,
        map_name: str = None,
        map_description: str = None,
        organization_name: str = None,
        reset_project_if_exists: bool = False,
        add_datums_if_exists: bool = False,
        shard_size: int = 1000,
        projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
        projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
        projection_spread: float = DEFAULT_PROJECTION_SPREAD,
        build_topic_model: bool = False,
        topic_label_field: str = None,
    ):
        '''
        Generates a map of the given embeddings.

        **Parameters:**

        * **embeddings** - An [N,d] numpy array containing the batch of N embeddings to add.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        * **id_field** - Specify your datas unique id field. ID fields can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        * **is_public** - Should this embedding map be public? Private maps can only be accessed by members of your organization.
        * **num_workers** - The number of workers to use when sending data.
        * **map_name** - A name for your map.
        * **map_description** - A description for your map.
        * **organization_name** - *(optional)* The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
        * **reset_project_if_exists** - If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        * **add_datums_if_exists** - If specifying an existing project and you want to add data to it, set this to true.
        * **shard_size** - The AtlasClient sends your data in shards to Atlas. A smaller shard_size sends more requests. Decrease the shard_size if you hit data size errors during upload.
        * **projection_n_neighbors** - *(optional)* The number of neighbors to use in the projection
        * **projection_epochs** - *(optional)* The number of epochs to use in the projection.
        * **projection_spread** - *(optional)* The effective scale of embedded points. Determines how clumped the map is.
        * **build_topic_model** - Builds a hierarchical topic model over your data to discover patterns.
        * **topic_label_field** - A text field to estimate topic labels from.

        **Returns:** A link to your map.
        '''
        if id_field is None:
            id_field = ATLAS_DEFAULT_ID_FIELD

        project_name = get_random_name()
        description = project_name
        index_name = get_random_name()

        if map_name:
            project_name = map_name
            index_name = map_name
        if map_description:
            description = map_description

        if data is None:
            data = [{} for _ in range(len(embeddings))]

        self._validate_map_data_inputs(colorable_fields=colorable_fields, id_field=id_field, data=data)

        project_id = self.create_project(
            project_name=project_name,
            description=description,
            unique_id_field=id_field,
            modality='embedding',
            is_public=is_public,
            organization_name=organization_name,
            reset_project_if_exists=reset_project_if_exists,
            add_datums_if_exists=add_datums_if_exists,
        )

        project = self._get_project_by_id(project_id=project_id)
        number_of_datums_before_upload = project['total_datums_in_project']

        # sends several requests to allow for threadpool refreshing. Threadpool hogs memory and new ones need to be created.
        logger.info("Uploading embeddings to Atlas.")

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
                    if number_of_datums_before_upload == 0:
                        logger.info("Deleting project due to failure in initial upload.")
                        self.delete_project(project_id=project_id)
                    raise e

        logger.info("Embedding upload succeeded.")

        # make a new index if there were no datums in the project before
        if number_of_datums_before_upload == 0:
            response = self.create_index(
                project_id=project_id,
                index_name=index_name,
                colorable_fields=colorable_fields,
                build_topic_model=build_topic_model,
                projection_n_neighbors=projection_n_neighbors,
                projection_epochs=projection_epochs,
                projection_spread=projection_spread,
                topic_label_field=topic_label_field,
            )
        else:
            # otherwise refresh the maps
            self.refresh_maps(project_id=project_id)
            return {'project_name': project_name, 'project_id': project_id}

        return dict(response)

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

        project = self._get_project_by_id(project_id=project_id)

        logger.info(f"Updating maps in project `{project['project_name']}`")

        return {'project_id': project_id}

    def map_text(
        self,
        data: List[Dict],
        indexed_field: str,
        id_field: str = None,
        is_public: bool = True,
        colorable_fields: list = [],
        num_workers: int = 10,
        map_name: str = None,
        map_description: str = None,
        organization_name: str = None,
        reset_project_if_exists: bool = False,
        add_datums_if_exists: bool = False,
        shard_size: int = 1000,
        projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
        projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
        projection_spread: float = DEFAULT_PROJECTION_SPREAD,
        build_topic_model: bool = False,
        multilingual: bool = False,
    ):
        '''
        Generates or updates a map of the given text.

        **Parameters:**

        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **indexed_field** - The name the data field corresponding to the text to be mapped.
        * **id_field** - Specify your datas unique id field. ID fields can be up 36 characters in length. If not specified, one will be created for you named `id_`.
        * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        * **is_public** - Should this embedding map be public? Private maps can only be accessed by members of your organization.
        * **num_workers** - The number of workers to use when sending data.
        * **map_name** - A name for your map.
        * **map_description** - A description for your map.
        * **organization_name** - *(optional)* The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
        * **reset_project_if_exists** - If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        * **add_datums_if_exists** - If specifying an existing project and you want to add data to it, set this to true.        * **shard_size** - The AtlasClient sends your data in shards to Atlas. A smaller shard_size sends more requests. Decrease the shard_size if you hit data size errors during upload.
        * **projection_n_neighbors** - *(optional)* The number of neighbors to use in the projection
        * **projection_epochs** - *(optional)* The number of epochs to use in the projection.
        * **projection_spread** - *(optional)* The effective scale of embedded points. Determines how clumped the map is.
        * **build_topic_model** - Builds a hierarchical topic model over your data to discover patterns.
        * **multilingual** - Should the map take language into account? If true, points from different languages but semantically similar text are close together.

        **Returns:** A link to your map.
        '''
        if id_field is None:
            id_field = ATLAS_DEFAULT_ID_FIELD

        project_name = get_random_name()
        description = project_name
        index_name = get_random_name()

        if map_name:
            project_name = map_name
            index_name = map_name
        if map_description:
            description = map_description

        self._validate_map_data_inputs(colorable_fields=colorable_fields, id_field=id_field, data=data)

        project_id = self.create_project(
            project_name=project_name,
            description=description,
            unique_id_field=id_field,
            modality='text',
            is_public=is_public,
            organization_name=organization_name,
            reset_project_if_exists=reset_project_if_exists,
            add_datums_if_exists=add_datums_if_exists,
        )

        project = self._get_project_by_id(project_id=project_id)
        number_of_datums_before_upload = project['total_datums_in_project']

        logger.info("Uploading text to Atlas.")

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
                    if number_of_datums_before_upload == 0:
                        logger.info("Deleting project due to failure in initial upload.")
                        self.delete_project(project_id=project_id)
                    raise e

        logger.info("Text upload succeeded.")

        # make a new index if there were no datums in the project before
        if number_of_datums_before_upload == 0:
            response = self.create_index(
                project_id=project_id,
                index_name=index_name,
                indexed_field=indexed_field,
                colorable_fields=colorable_fields,
                build_topic_model=build_topic_model,
                projection_n_neighbors=projection_n_neighbors,
                projection_epochs=projection_epochs,
                projection_spread=projection_spread,
                multilingual=multilingual,
            )
            return dict(response)
        else:
            # otherwise refresh the maps
            self.refresh_maps(project_id=project_id)
            return {'project_name': project_name, 'project_id': project_id}

        return dict(response)

    def delete_project(self, project_id: str):
        '''
        Deletes an atlas project with all associated metadata.

        **Parameters:**

        * **project_id** - The id of the project you want to delete.
        '''
        organization = self._get_current_users_main_organization()
        organization_name = organization['nickname']

        project = self._get_project_by_id(project_id=project_id)

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

        total_datums = project['total_datums_in_project']
        if project['insert_update_delete_lock']:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT

        def download_shard(offset, check_access=False):
            response = requests.get(
                self.atlas_api_path + f"/v1/project/data/get/embedding/{project_id}/{atlas_index_id}/{offset}/{limit}",
                headers=self.header,
            )

            if response.status_code != 200:
                raise Exception(response.json())

            if check_access:
                return
            try:
                content = response.json()

                shard_name = '{}_{}_{}.pkl'.format(atlas_index_id, offset, offset + limit)
                shard_path = os.path.join(output_dir, shard_name)
                with open(shard_path, 'wb') as f:
                    pickle.dump(content, f)

            except Exception as e:
                logger.error('Shard {} download failed with error: {}'.format(shard_name, e))

        download_shard(0, check_access=True)

        with tqdm(total=total_datums // limit) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(download_shard, cur_offset): cur_offset
                    for cur_offset in range(0, total_datums, limit)
                }
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
            self.atlas_api_path + f"/v1/project/{project_id}",
            headers=self.header,
        )
        project = response.json()
        if project['insert_update_delete_lock']:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT
        while True:
            response = requests.get(
                self.atlas_api_path + f"/v1/project/data/get/embedding/{project_id}/{atlas_index_id}/{offset}/{limit}",
                headers=self.header,
            )
            if response.status_code != 200:
                raise Exception(response.json()['detail'])

            content = response.json()
            if len(content['datum_ids']) == 0:
                break
            offset += len(content['datum_ids'])

            yield content['datum_ids'], content['embeddings']

    def get_nearest_neighbors(self, atlas_index_id: str, queries: np.array, k: int):
        '''
        Returns the nearest neighbors and the distances associated with a set of vector queries

        Args:
            atlas_index_id: the atlas index to use for the search
            queries: a 2d numpy array where each row corresponds to a query vetor
            k: the number of neighbors to return for each point

        Returns:
            A dictionary with the following information:
                neighbors: A set of ids corresponding to the nearest neighbors of each query
                distances: A set of distances between each query and its neighbors
        '''

        if queries.ndim != 2:
            raise ValueError(
                'Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).'
            )

        bytesio = io.BytesIO()
        np.save(bytesio, queries)

        status = 0
        retries = 0
        while status != 200 and retries < 10:
            response = requests.post(
                self.atlas_api_path + "/v1/project/data/get/embedding/query",
                headers=self.header,
                json={
                    'atlas_index_id': atlas_index_id,
                    'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                    'k': k,
                },
            )
            status = response.status_code
            retries += 1

        if retries == 10:
            raise AssertionError('Could not get response from server')

        return response.json()
