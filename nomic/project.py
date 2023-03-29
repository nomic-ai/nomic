import base64
import concurrent
import concurrent.futures
import io
import json
import os
import pickle
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Union
from typing import TYPE_CHECKING

from contextlib import contextmanager

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
import requests
from loguru import logger
from pyarrow import compute as pc
from pyarrow import feather, ipc
from pydantic import BaseModel, Field
from tqdm import tqdm


# Mainly for type checking
try:
    from pandas import DataFrame
    import pandas as pd
except ImportError:
    pd = None
    DataFrame = None

import nomic

from .cli import refresh_bearer_token, validate_api_http_response
from .settings import *
from .utils import assert_valid_project_id, get_object_size_in_bytes
from .data_inference import convert_pyarrow_schema_for_atlas

class AtlasUser:
    def __init__(self):
        self.credentials = refresh_bearer_token()

class AtlasClass(object):
    def __init__(self):
        '''
        Initializes the Atlas client.
        '''

        if self.credentials['tenant'] == 'staging':
            api_hostname = 'staging-api-atlas.nomic.ai'
            web_hostname = 'staging-atlas.nomic.ai'
        elif self.credentials['tenant'] == 'production':
            api_hostname = 'api-atlas.nomic.ai'
            web_hostname = 'atlas.nomic.ai'
        else:
            raise ValueError("Invalid tenant.")

        self.atlas_api_path = f"https://{api_hostname}"
        self.web_path = f"https://{web_hostname}"

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
                logger.warning(str(response))
                logger.info("Your authorization token is no longer valid.")
        else:
            raise ValueError(
                "Could not find an authorization token. Run `nomic login` to authorize this client with the Nomic API."
            )

    @property
    def credentials(self):
        return refresh_bearer_token()

    def _get_current_user(self):
        api_base_path = self.atlas_api_path
        if self.atlas_api_path.startswith('https://api-atlas.nomic.ai'):
            api_base_path = "https://no-cdn-api-atlas.nomic.ai"

        response = requests.get(
            api_base_path + "/v1/user",
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

    def _delete_project_by_id(self, project_id):
        response = requests.post(
            self.atlas_api_path + "/v1/project/remove",
            headers=self.header,
            json={'project_id': project_id},
        )

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
            raise Exception(f"Could not access project with id {project_id}: {response.text}")

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
            raise Exception(f'Could not access job state: {response.text}')

        return response.json()

    def _validate_and_correct_arrow_upload(
        self, data: pa.Table, project: "AtlasProject"
    ) -> pa.Table:
        '''
        Private method. validates upload data against the project arrow schema, and associated other checks.

        1. If unique_id_field is specified, validates that each datum has that field. If not, adds it and then notifies the user that it was added.        

        Args:
            data: an arrow table.
            project: the atlas project you are validating the data for.

        Returns:

        '''
        if not isinstance(data, pa.Table):
            raise Exception("Invalid data type for upload: {}".format(type(data)))

        if project.meta['modality'] == 'text':
            if "_embeddings" in data:
                msg = "Can't add embeddings to a text project."
                raise ValueError(msg)
        if project.meta['modality'] == 'embedding':
            if "_embeddings" not in data.column_names:
                msg = "Must include embeddings in embedding project upload."
                raise ValueError(msg)

        if project.id_field == ATLAS_DEFAULT_ID_FIELD and not ATLAS_DEFAULT_ID_FIELD in data.column_names:
            data = data.append_column(ATLAS_DEFAULT_ID_FIELD, pa.array([str(uuid.uuid4()) for _ in range(len(data))]))

        if project.schema is None:
            project._schema = convert_pyarrow_schema_for_atlas(data.schema)
        # Reformat to match the schema of the project.
        # This includes shuffling the order around if necessary,
        # filling in nulls, etc.
        reformatted = {}
        for field in project.schema:
            if field.name in data.column_names:                
                # Allow loss of precision in dates and ints, etc.
                reformatted[field.name] = data[field.name].cast(field.type, safe=False)
            else:                
                raise KeyError(f"Field {field.name} present in table schema not found in data. Present fields: {data.column_names}")
            if pa.types.is_string(field.type):
                # Ugly temporary measures
                if data[field.name].null_count > 0:
                    logger.warning(f"Replacing {data[field.name].null_count} null values for field {field.name} with string 'null'. This behavior will change in a future version.")
                    reformatted[field.name] = pc.fill_null(reformatted[field.name], "null")
                if pc.any(pc.equal(pc.binary_length(reformatted[field.name]), 0)):
                    mask = pc.equal(pc.binary_length(reformatted[field.name]), 0).combine_chunks()
                    assert pa.types.is_boolean(mask.type)
                    reformatted[field.name] = pc.replace_with_mask(reformatted[field.name], mask, "null")
        for field in data.schema:
            if not field.name in reformatted:
                if field.name == "_embeddings":
                    reformatted['_embeddings'] = data['_embeddings']
                else:
                    logger.warning(f"Field {field.name} present in data, but not found in table schema. Ignoring.")
        data = pa.Table.from_pydict(reformatted, schema=project.schema)
        
        if project.meta['insert_update_delete_lock']:
            raise Exception("Project is currently indexing and cannot ingest new datums. Try again later.")

        # The following two conditions should never occur given the above, but just in case...
        assert project.id_field in data.column_names, f"Upload does not contain your specified id_field"

        if not pa.types.is_string(data[project.id_field].type):
            logger.warning(f"id_field is not a string. Converting to string from {data[project.id_field].type}")
            data = data.drop([project.id_field]).append_column(project.id_field, data[project.id_field].cast(pa.string()))

        if data[project.id_field].null_count > 0:            
            raise ValueError(f"{project.id_field} must not contain null values, but {data[project.id_field].null_count} found.")
        
        for key in data.column_names:
            if key.startswith('_'):
                if key == '_embeddings':
                    continue
                raise ValueError('Metadata fields cannot start with _')

        if pc.max(pc.utf8_length(data[project.id_field])).as_py() > 36:
            first_match = data.filter(data[project.id_field].utf8_length() > 36).to_pylist()[0]
            raise ValueError(
                f"The id_field {first_match} is greater than 36 characters. Atlas does not support id_fields longer than 36 characters."
            )
        return data
    
    def _get_organization(self, organization_name=None, organization_id=None) -> Tuple[str, str]:
        '''
        Gets an organization by either it's name or id.

        Args:
            organization_name: the name of the organization
            organization_id: the id of the organization

        Returns:
            The organization_name and organization_id if one was found.

        '''

        if organization_name is None:
            if organization_id is None: #default to current users organization (the one with their name)
                organization = self._get_current_users_main_organization()
                organization_name = organization['nickname']
                organization_id = organization['organization_id']
            else:
                raise NotImplementedError("Getting organization by a specific ID is not yet implemented.")

        else:
            organization_id_request = requests.get(
                self.atlas_api_path + f"/v1/organization/search/{organization_name}", headers=self.header
            )
            if organization_id_request.status_code != 200:
                user = self._get_current_user()
                users_organizations = [org['nickname'] for org in user['organizations']]
                raise Exception(
                    f"No such organization exists: {organization_name}. You have access to the following organizations: {users_organizations}"
                )
            organization_id = organization_id_request.json()['organization_id']

        return organization_name, organization_id




    def _get_existing_project_by_name(self, project_name, organization_name) -> Dict:
        '''
        Utility method for instantiating an AtlasProject.
        Retrieves an existing project by name in a given organization. Fail
        Args:
            project_name: the project name
            organization_name: the organization name

        Returns:
            A dictionary containing the project_id, organization_id and organization_name

        '''

        # check if this project already exists.
        response = requests.post(
            self.atlas_api_path + "/v1/project/search/name",
            headers=self.header,
            json={'organization_name': organization_name, 'project_name': project_name},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to find project: {response.text}")
        search_results = response.json()['results']

        if search_results:
            existing_project = search_results[0]
            existing_project_id = existing_project['id']
            return {
                'project_id': existing_project_id,
                'organization_name': existing_project['owner'],
            }

        organization_name, organization_id = self._get_organization(organization_name=organization_name)
        return {'organization_id': organization_id, 'organization_name': organization_name}

class AtlasIndex:
    """
    An AtlasIndex represents a single view of an Atlas Project at a point in time.

    An AtlasIndex typically contains one or more *projections* which are 2D representations of
    the points in the index that you can browse online.
    """

    def __init__(self, atlas_index_id, name, indexed_field, projections):
        '''Initializes an Atlas index. Atlas indices organize data and store views of the data as maps.'''
        self.id = atlas_index_id
        self.name = name
        self.indexed_field = indexed_field
        self.projections = projections

    def _repr_html_(self):
        return '<br>'.join([d._repr_html_() for d in self.projections])

class AtlasProjection:
    '''
    Manages operations on maps such as text/vector search.
    This class should not be instantiated directly.
    Instead instantiate an AtlasProject and use the project.indices or get_map method to retrieve an AtlasProjection.
    '''


    def __init__(self, project : "AtlasProject", atlas_index_id: str, projection_id: str, name):
        """
        Creates an AtlasProjection.
        """
        self.project = project
        self.id = projection_id
        self.atlas_index_id = atlas_index_id
        self.projection_id = projection_id
        self.name = name

    @property
    def map_link(self):
        '''
        Retrieves a map link.
        '''
        return f"{self.project.web_path}/map/{self.project.id}/{self.id}"

    @property
    def _status(self):
        response = requests.get(
            self.project.atlas_api_path + f"/v1/project/index/job/progress/{self.atlas_index_id}",
            headers=self.project.header,
        )
        if response.status_code != 200:
            raise Exception(response.text)

        content = response.json()
        return content

    def __str__(self):
        return f"{self.name}: {self.map_link}"

    def __repr__(self):
        return self.__str__()

    def _iframe(self):
        return f"""
        <iframe class="iframe" id="iframe{self.id}" allow="clipboard-read; clipboard-write" src="{self.map_link}">
        </iframe>

        <style>
            .iframe {{
                /* vh can be **very** large in vscode ipynb. */
                height: min(75vh, 66vw);
                width: 100%;
            }}
        </style>
        """
    
    def _embed_html(self):
        return f"""<script>
            destroy = function() {{
                document.getElementById("iframe{self.id}").remove()
            }}
        </script>
        <div class="actions">
            <div id="hide" class="action" onclick="destroy()">Hide embedded project</div>
            <div class="action" id="out">
                <a href="{self.map_link}" target="_blank">Explore on atlas.nomic.ai</a>
            </div>
        </div>
        {self._iframe()}
        <style>
            .actions {{
              display: block;
            }}
            .action {{
              min-height: 18px;
              margin: 5px;
              transition: all 500ms ease-in-out;
            }}
            .action:hover {{
              cursor: pointer;
            }}
            #hide:hover::after {{
                content: " X";
            }}
            #out:hover::after {{
                content: "";
            }}
        </style>
        """

    def _repr_html_(self):
        # Don't make an iframe if the project is locked.
        state = self._status['index_build_stage']
        if state != 'Completed':
            return f"""Atlas Projection {self.name}. Status {state}. <a target="_blank" href="{self.map_link}">view online</a>"""
        return f"""
            <h3>Project: {self.name}</h3>
            {self._embed_html()}
            """
    
    def _download_feather(self, dest: str = "tiles"):
        '''
        Downloads the feather tree.
        Args:
            dest: the destination to download the quadtree.

        Returns:
            A list containing all quadtiles downloads
        '''
        dest = Path(dest)
        root = f'{self.project.atlas_api_path}/v1/project/public/{self.project.id}/index/projection/{self.id}/quadtree/'
        quads = [f'0/0/0']
        all_quads = []
        while len(quads) > 0:
            quad = quads.pop(0) + ".feather"
            all_quads.append(quad)
            path = dest / quad
            if not path.exists():
                data = requests.get(root + quad)
                readable = io.BytesIO(data.content)
                readable.seek(0)
                tb = feather.read_table(readable)
                path.parent.mkdir(parents=True, exist_ok=True)
                feather.write_feather(tb, path)
            schema = ipc.open_file(path).schema
            kids = schema.metadata.get(b'children')
            children = json.loads(kids)
            quads.extend(children)
        return all_quads

    def download_embeddings(self, save_directory: str, num_workers: int = 10) -> bool:
        '''
        Downloads a mapping from datum_id to embedding in shards to the provided directory

        Args:
            save_directory: The directory to save your embeddings.
        Returns:
            True on success


        '''
        self.project._latest_project_state()

        total_datums = self.project.total_datums
        if self.project.is_locked:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT

        def download_shard(offset, check_access=False):
            response = requests.get(
                self.project.atlas_api_path + f"/v1/project/data/get/embedding/{self.project.id}/{self.atlas_index_id}/{offset}/{limit}",
                headers=self.project.header,
            )

            if response.status_code != 200:
                raise Exception(response.text)

            if check_access:
                return
            try:
                content = response.json()

                shard_name = '{}_{}_{}.pkl'.format(self.atlas_index_id, offset, offset + limit)
                shard_path = os.path.join(save_directory, shard_name)
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

        return True


    def get_embedding_iterator(self) -> Iterable[Tuple[str, str]]:
        '''
        Iterate through embeddings of your datums.

        Returns:
            A iterable mapping datum ids to their embeddings.

        '''

        if self.is_locked:
            raise Exception('Project is locked! Please wait until the project is unlocked to download embeddings')

        offset = 0
        limit = EMBEDDING_PAGINATION_LIMIT
        while True:
            response = requests.get(
                self.atlas_api_path + f"/v1/project/data/get/embedding/{self.id}/{self.atlas_index_id}/{offset}/{limit}",
                headers=self.header,
            )
            if response.status_code != 200:
                raise Exception(response.text)

            content = response.json()
            if len(content['datum_ids']) == 0:
                break
            offset += len(content['datum_ids'])

            yield content['datum_ids'], content['embeddings']

    def vector_search(self, queries: np.array = None, ids: List[str] = None, k: int = 5) -> Dict[str, List]:
        '''
        Performs vector similarity search over data points on your map.
        If ids is specified, receive back the most similar data ids in vector space to your input ids.
        If queries is specified, receive back the data ids with representations most similar to the query vectors.

        You should not specify both queries and ids.

        Args:
            queries: a 2d numpy array where each row corresponds to a query vector
            ids: a list of
            k: the number of closest data points (neighbors) to return for each input query/data id
        Returns:
            A tuple with two elements containing the following information:
                neighbors: A set of ids corresponding to the nearest neighbors of each query
                distances: A set of distances between each query and its neighbors
        '''

        if queries is None and ids is None:
            raise ValueError('You must specify either a list of datum `ids` or numpy array of `queries` but not both.')

        max_k = 128
        max_queries = 256
        if k > max_k:
            raise Exception(f"Cannot query for more than {max_k} nearest neighbors. Set `k` to {max_k} or lower")

        if ids is not None:
            if len(ids) > max_queries:
                raise Exception(f"Max ids per query is {max_queries}. You sent {len(ids)}.")
        if queries is not None:
            if not isinstance(queries, np.ndarray):
                raise Exception("`queries` must be an instance of np.array.")
            if queries.shape[0] > max_queries:
                raise Exception(f"Max vectors per query is {max_queries}. You sent {queries.shape[0]}.")

        if queries is not None:
            if queries.ndim != 2:
                raise ValueError('Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).')

            bytesio = io.BytesIO()
            np.save(bytesio, queries)

        if queries is not None:
            response = requests.post(
                self.project.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_embedding",
                headers=self.project.header,
                json={'atlas_index_id': self.atlas_index_id,
                      'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                      'k': k},
            )
        else:
            response = requests.post(
                self.project.atlas_api_path + "/v1/project/data/get/nearest_neighbors/by_id",
                headers=self.project.header,
                json={'atlas_index_id': self.atlas_index_id,
                      'datum_ids': ids,
                      'k': k},
            )


        if response.status_code == 500:
            raise Exception('Cannot perform vector search on your map at this time. Try again later.')

        if response.status_code != 200:
            raise Exception(response.text)

        response = response.json()

        return response['neighbors'], response['distances']

    def get_topic_data(self) -> List:
        '''
        Retrieves metadata about a maps pre-computed topics

        Returns:
            A dictionary of metadata for each topic in the model

        '''
        response = requests.get(
            self.project.atlas_api_path + "/v1/project/{}/index/projection/{}".format(self.project.meta['id'], self.projection_id),
            headers=self.project.header,
        )
        topics = json.loads(response.text)['topic_models'][0]['features']
        topic_data = [e['properties'] for e in topics]
        return topic_data

    def get_topic_density(self, time_field: str, start: datetime, end: datetime):
        '''
        Counts the number of datums in each topic within a window

        Args:
            time_field: Your metadata field containing isoformat timestamps
            start: A datetime object for the window start
            end: A datetime object for the window end

        Returns:
            List[{topic: str, count: int}] - A list of {topic, count} dictionaries, sorted from largest count to smallest count
        '''
        response = requests.post(
            self.project.atlas_api_path + "/v1/project/{}/topic_density".format(self.atlas_index_id),
            headers=self.project.header,
            json={'start': start.isoformat(),
                  'end': end.isoformat(),
                  'time_field': time_field},
        )
        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()


    def vector_search_topics(self, queries: np.array, k: int = 32, depth: int = 3) -> Dict:
        '''
        Returns the topics best associated with each vector query

        Args:
            queries: a 2d numpy array where each row corresponds to a query vector
            k: (Default 32) the number of neighbors to use when estimating the posterior
            depth: (Default 3) the topic depth at which you want to search

        Returns:
            A dict of {topic: posterior probability} for each query
        '''

        if queries.ndim != 2:
            raise ValueError('Expected a 2 dimensional array. If you have a single query, we expect an array of shape (1, d).')

        bytesio = io.BytesIO()
        np.save(bytesio, queries)

        response = requests.post(
            self.project.atlas_api_path + "/v1/project/data/get/embedding/topic",
            headers=self.project.header,
            json={'atlas_index_id': self.atlas_index_id,
                  'queries': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                  'k': k,
                  'depth': depth},
        )

        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()


    def _get_atoms(self, ids: List[str]) -> List[Dict]:
        '''
        Retrieves atoms by id

        Args:
            ids: list of atom ids

        Returns:
            A dictionary containing the resulting atoms, keyed by atom id.

        '''

        if not isinstance(ids, list):
            raise ValueError("You must specify a list of ids when getting data.")

        response = requests.post(
            self.project.atlas_api_path + "/v1/project/atoms/get",
            headers=self.project.header,
            json={'project_id': self.project.id, 'index_id': self.atlas_index_id, 'atom_ids': ids},
        )

        if response.status_code == 200:
            return response.json()['atoms']
        else:
            raise Exception(response.text)

    def get_tags(self) -> Dict[str, List[str]]:
        '''
        Retrieves back all tags made in the web browser for a specific project and map.

        Returns:
            A dictionary mapping datum ids to tags.
        '''
        # now get the tags
        datums_and_tags = requests.post(
            self.project.atlas_api_path + '/v1/project/tag/read/all_by_datum',
            headers=self.project.header,
            json={
                'project_id': self.project.id,
            },
        ).json()['results']

        label_to_datums = {}
        for item in datums_and_tags:
            for label in item['labels']:
                if label not in label_to_datums:
                    label_to_datums[label] = []
                label_to_datums[label].append(item['datum_id'])
        return label_to_datums

    def tag(self, ids: List[str], tags: List[str]):
        '''

        Args:
            ids: The datum ids you want to tag
            tags: A list containing the tags you want to apply to these data points.

        '''
        assert isinstance(ids, list), 'ids must be a list of strings'
        assert isinstance(tags, list), 'tags must be a list of strings'

        colname = json.dumps({'project_id': self.project.id, 'atlas_index_id': self.atlas_index_id, 'type': 'datum_id', 'tags': tags})
        payload_table = pa.table([pa.array(ids, type=pa.string())], [colname])
        buffer = io.BytesIO()
        writer = ipc.new_file(buffer, payload_table.schema, options=ipc.IpcWriteOptions(compression='zstd'))
        writer.write_table(payload_table)
        writer.close()
        payload = buffer.getvalue()

        headers = self.project.header.copy()
        headers['Content-Type'] = 'application/octet-stream'
        response = requests.post(self.project.atlas_api_path + "/v1/project/tag/add", headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception("Failed to add tags")

    def remove_tags(self, ids: List[str], tags: List[str], delete_all: bool = False) -> bool:
        '''
        Deletes the specified tags from the given datum_ids.

        Args:
            ids: The datum_ids to delete tags from.
            tags: The list of tags to delete from the data points. Each tag will be applied to all data points in `ids`.
            delete_all: If true, ignores datum_ids and deletes all specified tags from all data points.

        Returns:
            True on success

        '''
        assert isinstance(ids, list), 'datum_ids must be a list of strings'
        assert isinstance(tags, list), 'tags must be a list of strings'

        colname = json.dumps({'project_id': self.project.id, 'atlas_index_id': self.atlas_index_id, 'type': 'datum_id', 'tags': tags, 'delete_all': delete_all})
        payload_table = pa.table([pa.array(ids, type=pa.string())], [colname])
        buffer = io.BytesIO()
        writer = ipc.new_file(buffer, payload_table.schema, options=ipc.IpcWriteOptions(compression='zstd'))
        writer.write_table(payload_table)
        writer.close()
        payload = buffer.getvalue()

        headers = self.project.header.copy()
        headers['Content-Type'] = 'application/octet-stream'
        response = requests.post(self.project.atlas_api_path + "/v1/project/tag/delete", headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception("Failed to delete tags")

class AtlasProject(AtlasClass):
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = 'A description for your map.',
        unique_id_field: Optional[str] = None,
        modality: Optional[str] = None,
        organization_name: Optional[str] = None,
        is_public: bool = True,
        project_id=None,
        reset_project_if_exists=False,
        add_datums_if_exists=True,
    ):

        """
        Creates or loads an Atlas project.
        Atlas projects store data (text, embeddings, etc) that you can organize by building indices.
        If the organization already contains a project with this name, it will be returned instead.

        **Parameters:**

        * **project_name** - The name of the project.
        * **description** - A description for the project.
        * **unique_id_field** - The field that uniquely identifies each datum. If a datum does not contain this field, it will be added and assigned a random unique ID.
        * **modality** - The data modality of this project. Currently, Atlas supports either `text` or `embedding` modality projects.
        * **organization_name** - The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user account's default organization.
        * **is_public** - Should this project be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.
        * **reset_project_if_exists** - If the requested project exists in your organization, will delete it and re-create it.
        * **project_id** - An alternative way to retrieve a project is by passing the project_id directly. This only works if a project exists.
        * **reset_project_if_exists** - If the specified project exists in your organization, reset it by deleting all of its data. This means your uploaded data will not be contextualized with existing data.
        * **add_datums_if_exists** - If specifying an existing project and you want to add data to it, set this to true.

        """
        assert name is not None or project_id is not None, "You must pass a name or project_id"

        super().__init__()

        if project_id is not None:
            self.meta = self._get_project_by_id(project_id)
            return

        if organization_name is None:
            organization_name = self._get_current_users_main_organization()['nickname']

        results = self._get_existing_project_by_name(project_name=name, organization_name=organization_name)

        if 'project_id' in results: #project already exists
            organization_name = results['organization_name']
            project_id = results['project_id']
            if reset_project_if_exists: #reset the project
                logger.info(
                    f"Found existing project `{name}` in organization `{organization_name}`. Clearing it of data by request."
                )
                self._delete_project_by_id(project_id=project_id)
                project_id = None
            elif not add_datums_if_exists: #prevent adding datums to existing project explicitly
                raise ValueError(
                    f"Project already exists with the name `{name}` in organization `{organization_name}`. "
                    f"You can add datums to it by settings `add_datums_if_exists = True` or reset it by specifying `reset_project_if_exist=True` on a new upload."
                )
            else:
                logger.info(
                    f"Loading existing project `{name}` from organization `{organization_name}`."
                )

        if project_id is None: #if there is no existing project, make a new one.

            if unique_id_field is None:
                unique_id_field = ATLAS_DEFAULT_ID_FIELD

                raise ValueError("You must specify a unique_id_field when creating a new project.")

            if modality is None:
                raise ValueError("You must specify a modality when creating a new project.")

            assert modality in ['text', 'embedding'], "Modality must be either `text` or `embedding`"
            assert name is not None

            project_id = self._create_project(
                project_name=name,
                description=description,
                unique_id_field=unique_id_field,
                modality=modality,
                organization_name=organization_name,
                is_public=is_public
            )


        self.meta = self._get_project_by_id(project_id=project_id)
        self._schema = None



    def delete(self):
        '''
        Deletes an atlas project with all associated metadata.
        '''
        organization = self._get_current_users_main_organization()
        organization_name = organization['nickname']

        logger.info(f"Deleting project `{self.name}` from organization `{organization_name}`")

        self._delete_project_by_id(project_id=self.id)

        return False

    def _create_project(
        self,
        project_name: str,
        description: Optional[str],
        unique_id_field: str,
        modality: str,
        organization_name: Optional[str] = None,
        is_public: bool = True
    ):
        '''
        Creates an Atlas project.
        Atlas projects store data (text, embeddings, etc) that you can organize by building indices.
        If the organization already contains a project with this name, it will be returned instead.

        **Parameters:**

        * **project_name** - The name of the project.
        * **description** - A description for the project.
        * **unique_id_field** - The field that uniquely identifies each datum. If a datum does not contain this field, it will be added and assigned a random unique ID.
        * **modality** - The data modality of this project. Currently, Atlas supports either `text` or `embedding` modality projects.
        * **organization_name** - The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user accounts default organization.
        * **is_public** - Should this project be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.

        **Returns:** project_id on success.

        '''

        organization_name, organization_id = self._get_organization(organization_name=organization_name)

        supported_modalities = ['text', 'embedding']
        if modality not in supported_modalities:
            msg = 'Tried to create project with modality: {}, but Atlas only supports: {}'.format(
                modality, supported_modalities
            )
            raise ValueError(msg)

        if unique_id_field is None:
            raise ValueError("You must specify a unique id field")
        logger.info(f"Creating project `{project_name}` in organization `{organization_name}`")
        if description is None:
            description = ""
        response = requests.post(
            self.atlas_api_path + "/v1/project/create",
            headers=self.header,
            json = {
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


    def _latest_project_state(self):
        '''
        Refreshes the project's state. Try to call this sparingly but use it when you need it.
        '''

        self.meta = self._get_project_by_id(self.id)
        return self

    @property
    def indices(self) -> List[AtlasIndex]:
        self._latest_project_state()
        output = []
        for index in self.meta['atlas_indices']:
            projections = []
            for projection in index['projections']:
                projection = AtlasProjection(
                    project=self, projection_id=projection['id'], atlas_index_id=index['id'], name=index['index_name']
                )
                projections.append(projection)
            index = AtlasIndex(atlas_index_id=index['id'], name=index['index_name'], indexed_field=index['indexed_field'], projections=projections)
            output.append(index)

        return output

    @property
    def projections(self) -> List[AtlasProjection]:
        output = []
        for index in self.indices:
            for projection in index.projections:
                output.append(projection)
        return output

    @property
    def maps(self) -> List[AtlasProjection]:
        return self.projections

    @property
    def id(self) -> str:
        '''The UUID of the project.'''
        return self.meta['id']

    @property
    def id_field(self) -> str:
        return self.meta['unique_id_field']

    @property
    def total_datums(self) -> int:
        '''The total number of data points in the project.'''
        return self.meta['total_datums_in_project']

    @property
    def modality(self) -> str:
        return self.meta['modality']

    @property
    def name(self) -> str:
        '''The name of the project.'''
        return self.meta['project_name']

    @property
    def description(self):
        return self.meta['description']

    @property
    def project_fields(self):
        return self.meta['project_fields']

    @property
    def is_locked(self) -> bool:
        self._latest_project_state()
        return self.meta['insert_update_delete_lock']
    
    @property
    def schema(self) -> Optional[pa.Schema]:
        if self._schema is not None:
            return self._schema
        if 'schema' in self.meta and self.meta['schema'] is not None:
            self._schema : pa.Schema = ipc.read_schema(io.BytesIO(base64.b64decode(self.meta['schema'])))
            return self._schema
        return None

    @property
    def is_accepting_data(self) -> bool:
        '''
        Checks if the project can accept data. Projects cannot accept data when they are being indexed.

        Returns:
            True if project is unlocked for data additions, false otherwise.
        '''
        return not self.is_locked

    @contextmanager
    def wait_for_project_lock(self):
        '''Blocks thread execution until project is in a state where it can ingest data.'''
        has_logged = False
        while True:
            if self.is_accepting_data:
                yield self
                break
            if not has_logged:
                logger.info(f"{self.name}: Waiting for Project Lock Release.")
                has_logged = True
            time.sleep(5)

    def get_map(self, name: str = None, atlas_index_id: str = None, projection_id: str = None) -> AtlasProjection:
        '''
        Retrieves a Map

        Args:
            name: The name of your map. This defaults to your projects name but can be different if you build multiple maps in your project.
            atlas_index_id: If specified, will only return a map if there is one built under the index with the id atlas_index_id.
            projection_id: If projection_id is specified, will only return a map if there is one built under the index with id projection_id.

        Returns:
            The map or a ValueError.
        '''

        indices = self.indices

        if atlas_index_id is not None:
            for index in indices:
                if index.id == atlas_index_id:
                    if len(index.projections) == 0:
                        raise ValueError(f"No map found under index with atlas_index_id='{atlas_index_id}'")
                    return index.projections[0]
            raise ValueError(f"Could not find a map with atlas_index_id='{atlas_index_id}'")

        if projection_id is not None:
            for index in indices:
                for projection in index.projections:
                    if projection.id == projection_id:
                        return projection
            raise ValueError(f"Could not find a map with projection_id='{atlas_index_id}'")

        if len(indices) == 0:
            raise ValueError("You have no maps built in your project")
        if len(indices) > 1 and name is None:
            raise ValueError("You have multiple maps in this project, specify a name.")

        if len(indices) == 1:
            if len(indices[0].projections) == 1:
                return indices[0].projections[0]

        for index in indices:
            if index.name == name:
                return index.projections[0]

        raise ValueError(f"Could not find a map named {name} in your project.")

    def create_index(
        self,
        name: str,
        indexed_field: str = None,
        colorable_fields: list = [],
        multilingual: bool = False,
        build_topic_model: bool = False,
        projection_n_neighbors: int = DEFAULT_PROJECTION_N_NEIGHBORS,
        projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
        projection_spread: float = DEFAULT_PROJECTION_SPREAD,
        topic_label_field: str = None,
        reuse_embeddings_from_index: str = None,
    ) -> AtlasProjection:
        '''
        Creates an index in the specified project.

        Args:
            name: The name of the index and the map.
            indexed_field: For text projects, name the data field corresponding to the text to be mapped.
            colorable_fields: The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
            multilingual: Should the map take language into account? If true, points from different languages but semantically similar text are close together.
            build_topic_model: Should a topic model be built?
            projection_n_neighbors: A projection hyperparameter
            projection_epochs: A projection hyperparameter
            projection_spread: A projection hyperparameter
            topic_label_field: A text field in your metadata to estimate topic labels from. Defaults to the indexed_field for text projects if not specified.
            reuse_embeddings_from_index: the name of the index to reuse embeddings from.

        Returns:
            The projection this index has built.

        '''

        self._latest_project_state()

        # for large projects, alter the default projection configurations.
        if self.total_datums >= 1_000_000:
            if (
                projection_epochs == DEFAULT_PROJECTION_EPOCHS
                and projection_n_neighbors == DEFAULT_PROJECTION_N_NEIGHBORS
            ):
                projection_n_neighbors = DEFAULT_LARGE_PROJECTION_N_NEIGHBORS
                projection_epochs = DEFAULT_LARGE_PROJECTION_EPOCHS

        if self.modality == 'embedding':
            build_template = {
                'project_id': self.id,
                'index_name': name,
                'indexed_field': None,
                'atomizer_strategies': None,
                'model': None,
                'colorable_fields': colorable_fields,
                'model_hyperparameters': None,
                'nearest_neighbor_index': 'HNSWIndex',
                'nearest_neighbor_index_hyperparameters': json.dumps({'space': 'l2', 'ef_construction': 100, 'M': 16}),
                'projection': 'NomicProject',
                'projection_hyperparameters': json.dumps(
                    {'n_neighbors': projection_n_neighbors, 'n_epochs': projection_epochs, 'spread': projection_spread}
                ),
                'topic_model_hyperparameters': json.dumps(
                    {'build_topic_model': build_topic_model, 'community_description_target_field': topic_label_field}
                ),
            }

        elif self.modality == 'text':

            #find the index id of the index with name reuse_embeddings_from_index
            reuse_embedding_from_index_id = None
            indices = self.indices
            if reuse_embeddings_from_index is not None:
                for index in indices:
                    if index.name == reuse_embeddings_from_index:
                        reuse_embedding_from_index_id = index.id
                        break
                if reuse_embedding_from_index_id is None:
                    raise Exception(f"Could not find the index '{reuse_embeddings_from_index}' to re-use from. Possible options are {[index.name for index in indices]}")

            if indexed_field is None:
                raise Exception("You did not specify a field to index. Specify an 'indexed_field'.")

            if indexed_field not in self.project_fields:
                raise Exception(f"Indexing on {indexed_field} not allowed. Valid options are: {self.project_fields}")

            model = 'NomicEmbed'
            if multilingual:
                model = 'NomicEmbedMultilingual'

            build_template = {
                'project_id': self.id,
                'index_name': name,
                'indexed_field': indexed_field,
                'atomizer_strategies': ['document', 'charchunk'],
                'model': model,
                'colorable_fields': colorable_fields,
                'reuse_atoms_and_embeddings_from': reuse_embedding_from_index_id,
                'model_hyperparameters': json.dumps(
                    {
                        'dataset_buffer_size': 1000,
                        'batch_size': 20,
                        'polymerize_by': 'charchunk',
                        'norm': 'both',
                    }
                ),
                'nearest_neighbor_index': 'HNSWIndex',
                'nearest_neighbor_index_hyperparameters': json.dumps({'space': 'l2', 'ef_construction': 100, 'M': 16}),
                'projection': 'NomicProject',
                'projection_hyperparameters': json.dumps(
                    {'n_neighbors': projection_n_neighbors, 'n_epochs': projection_epochs, 'spread': projection_spread}
                ),
                'topic_model_hyperparameters': json.dumps(
                    {'build_topic_model': build_topic_model, 'community_description_target_field': indexed_field}
                ),
            }

        response = requests.post(
            self.atlas_api_path + "/v1/project/index/create",
            headers=self.header,
            json=build_template,
        )
        if response.status_code != 200:
            logger.info('Create project failed with code: {}'.format(response.status_code))
            logger.info('Additional info: {}'.format(response.text))
            raise Exception(response.json()['detail'])

        job_id = response.json()['job_id']

        job = requests.get(
            self.atlas_api_path + f"/v1/project/index/job/{job_id}",
            headers=self.header,
        ).json()

        index_id = job['index_id']

        try:
            projection = self.get_map(atlas_index_id=index_id)
        except ValueError:
            # give some delay
            time.sleep(5)
            try:
                projection = self.get_map(atlas_index_id=index_id)
            except ValueError:
                projection = None

        if projection is None:
            logger.warning(
                "Could not find a map being built for this project. See atlas.nomic.ai/dashboard for map status."
            )
        logger.info(f"Created map `{projection.name}` in project `{self.name}`: {projection.map_link}")
        return projection

    def __repr__(self):
        m = self.meta
        return f"AtlasProject: <{m}>"

    def _repr_html_(self):
        self._latest_project_state()
        m = self.meta
        html = f"""
            <strong><a href="https://atlas.nomic.ai/dashboard/project/{m['id']}">{m['project_name']}</strong></a>
            <br>
            {m['description']} {m['total_datums_in_project']} datums inserted.
            <br>
            {len(m['atlas_indices'])} index built.
            """
        complete_projections = []
        if len(self.projections) >= 1:
            html += "<br><strong>Projections</strong>\n"
            html += "<ul>\n"
            for projection in self.projections:
                state = projection._status['index_build_stage']
                if state == 'Completed':
                    complete_projections.append(projection)
                html += f"""<li>{projection.name}. Status {state}. <a target="_blank" href="{projection.map_link}">view online</a></li>"""   
            html += "</ul>"
        if len(complete_projections) >= 1:
            # Display most recent complete projection.
            html += "<hr>"
            html += complete_projections[-1]._embed_html()
        return html
    
    def __str__(self):
        return "\n".join([str(projection) for index in self.indices for projection in index.projections])

    def get_data(self, ids: List[str]) -> List[Dict]:
        '''
        Retrieve the contents of the data given ids

        Args:
            ids: a list of datum ids

        Returns:
            A list of dictionaries corresponding

        '''

        if not isinstance(ids, list):
            raise ValueError("You must specify a list of ids when getting data.")
        if isinstance(ids[0], list):
            raise ValueError("You must specify a list of ids when getting data, not a nested list.")
        response = requests.post(
            self.atlas_api_path + "/v1/project/data/get",
            headers=self.header,
            json={'project_id': self.id, 'datum_ids': ids},
        )

        if response.status_code == 200:
            return [item for item in response.json()['datums']]
        else:
            raise Exception(response.text)

    def delete_data(self, ids: List[str]) -> bool:
        '''
        Deletes the specified datums from the project.

        Args:
            ids: A list of datum ids to delete

        Returns:
            True if data deleted successfully.

        '''
        if not isinstance(ids, list):
            raise ValueError("You must specify a list of ids when deleting datums.")

        response = requests.post(
            self.atlas_api_path + "/v1/project/data/delete",
            headers=self.header,
            json={'project_id': self.id, 'datum_ids': ids},
        )

        if response.status_code == 200:
            return True
        else:
            raise Exception(response.text)

    def add_text(
        self,
        data = Union[DataFrame, List[Dict], pa.Table],
        pbar=None,
        shard_size=None,
        num_workers=None
    ):
        """
        Add text data to the project.
        data: A pandas DataFrame, a list of python dictionaries, or a pyarrow Table matching the project schema.
        pbar: (Optional). A tqdm progress bar to display progress.
        """
        if shard_size is not None or num_workers is not None:
            raise DeprecationWarning("shard_size and num_workers are deprecated.")
        if DataFrame is not None and isinstance(data, DataFrame):
            data = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            data = pa.Table.from_pylist(data)
        elif not isinstance(data, pa.Table):
            raise ValueError("Data must be a pandas DataFrame, list of dictionaries, or a pyarrow Table.")
        self._add_data(data, pbar=pbar)

    def add_embeddings(
            self,
            data : Union[DataFrame, List[Dict], pa.Table, None],
            embeddings: np.array,
            pbar = None,
            shard_size=None, num_workers=None
    ):
        """
        Add data, with associated embeddings, to the project.
        
        Args:
            data: A pandas DataFrame, list of dictionaries, or pyarrow Table matching the project schema.
            embeddings: A numpy array of embeddings: each row corresponds to a row in the table.
            pbar: (Optional). A tqdm progress bar to update.
        """
        
        """
        # TODO: validate embedding size.
        assert embeddings.shape[1] == self.embedding_size, "Embedding size must match the embedding size of the project."
        """
        if shard_size is not None:
            raise DeprecationWarning("shard_size is deprecated and no longer has any effect")
        if num_workers is not None:
            raise DeprecationWarning("num_workers is deprecated and no longer has any effect")
        assert type(embeddings) == np.ndarray, "Embeddings must be a numpy array."
        assert len(embeddings.shape) == 2, "Embeddings must be a 2D numpy array."
        assert len(data) == embeddings.shape[0], "Data and embeddings must have the same number of rows."
        assert len(data) > 0, "Data must have at least one row."
        
        tb: pa.Table

        if DataFrame is not None and isinstance(data, DataFrame):
            tb = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            tb = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            tb = data
        else:
            raise ValueError(f"Data must be a pandas DataFrame, list of dictionaries, or a pyarrow Table, not {type(data)}")
        
        del data

        # Add embeddings to the data.
        embeddings = embeddings.astype(np.float16)

        # Fail if any embeddings are NaN or Inf.
        assert not np.isnan(embeddings).any(), "Embeddings must not contain NaN values."
        assert not np.isinf(embeddings).any(), "Embeddings must not contain Inf values."

        pyarrow_embeddings = pa.FixedSizeListArray.from_arrays(embeddings.reshape((-1)), embeddings.shape[1])

        data_with_embeddings = tb.append_column("_embeddings", pyarrow_embeddings)

        self._add_data(data_with_embeddings, pbar=pbar)

    def _add_data(
        self,
        data: pa.Table,
        pbar=None,
    ):
        '''
        Low level interface to upload an Arrow Table. Users should generally call 'add_text' or 'add_embeddings.'

        Args:
            data: A pyarrow Table that will be cast to the project schema.
            pbar: A tqdm progress bar to update.
        Returns:
            None
        '''

        # Exactly 10 upload workers at a time.

        num_workers = 10

        # Each worker currently is too slow beyond a shard_size of 5000

        # The heuristic here is: Never let shards be more than 5,000 items,
        # OR more than 4MB uncompressed. Whichever is smaller.

        bytesize = data.nbytes
        nrow = len(data)

        shard_size = 5000
        n_chunks = int(np.ceil(nrow / shard_size))
        # Chunk into 4MB pieces. These will probably compress down a bit.
        if bytesize / n_chunks > 4_000_000:
            shard_size = int(np.ceil(nrow / (bytesize / 4_000_000)))
        

        data = self._validate_and_correct_arrow_upload(
                data=data,
                project=self
            )

        upload_endpoint = "/v1/project/data/add/arrow"

        # Actually do the upload
        def send_request(i):
            data_shard = data.slice(i, shard_size)
            with io.BytesIO() as buffer:
                data_shard = data_shard.replace_schema_metadata({'project_id': self.id})
                feather.write_feather(data_shard, buffer, compression = 'zstd', compression_level = 6)
                buffer.seek(0)

                response = requests.post(
                    self.atlas_api_path + upload_endpoint,
                    headers=self.header,
                    data = buffer,
                    json={'project_id': self.id},
                )
                return response

        # if this method is being called internally, we pass a global progress bar
        close_pbar = False
        if pbar is None:
            close_pbar = True
            pbar = tqdm(total=int(len(data)) // shard_size)
        failed = 0
        succeeded = 0
        errors_504 = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(send_request, i): i for i in range(0, len(data), shard_size)}

            while futures:
                # check for status of the futures which are currently working
                done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                # process any completed futures
                for future in done:
                    response = future.result()
                    if response.status_code != 200:
                        try:
                            logger.error(f"Shard upload failed: {response.text}")
                            if 'more datums exceeds your organization limit' in response.json():
                                return False
                            if 'Project transaction lock is held' in response.json():
                                raise Exception(
                                    "Project is currently indexing and cannot ingest new datums. Try again later."
                                )
                            if 'Insert failed due to ID conflict' in response.json():
                                continue
                        except (requests.JSONDecodeError, json.decoder.JSONDecodeError):
                            if response.status_code == 413:
                                # Possibly split in two and retry?
                                logger.error("Shard upload failed: you are sending meta-data that is too large.")
                                pbar.update(1)
                                response.close()
                                failed += shard_size
                            elif response.status_code == 504:
                                errors_504 += shard_size
                                start_point = futures[future]
                                logger.debug(
                                    f"{self.name}: Connection failed for records {start_point}-{start_point + shard_size}, retrying."
                                )
                                failure_fraction = errors_504 / (failed + succeeded + errors_504)
                                if failure_fraction > 0.5 and errors_504 > shard_size * 3:
                                    raise RuntimeError(
                                        f"{self.name}: Atlas is under high load and cannot ingest datums at this time. Please try again later."
                                    )
                                new_submission = executor.submit(send_request, start_point)
                                futures[new_submission] = start_point
                                response.close()
                            else:
                                logger.error(f"{self.name}: Shard upload failed: {response}")
                                failed += shard_size
                                pbar.update(1)
                                response.close()
                    else:
                        # A successful upload.
                        succeeded += shard_size
                        pbar.update(1)
                        response.close()

                    # remove the now completed future
                    del futures[future]

        # close the progress bar if this method was called with no external progresbar
        if close_pbar:
            pbar.close()

        if failed:
            logger.warning(f"Failed to upload {failed} datums")
        if close_pbar:
            if failed:
                logger.warning("Upload partially succeeded.")
            else:
                logger.info("Upload succeeded.")

    def update_maps(self,
                    data: List[Dict],
                    embeddings: Optional[np.array]=None,
                    shard_size: int = 1000,
                    num_workers: int = 10):
        '''
        Utility method to update a projects maps by adding the given data.

        Args:
            data: An [N,] element list of dictionaries containing metadata for each embedding.
            embeddings: An [N, d] matrix of embeddings for updating embedding projects. Leave as None to update text projects.
            shard_size: Data is uploaded in parallel by many threads. Adjust the number of datums to upload by each worker.
            num_workers: The number of workers to use when sending data.

        '''

        # Validate data
        if self.modality == 'embedding' and embeddings is None:
            msg = 'Please specify embeddings for updating an embedding project'
            raise ValueError(msg)

        if self.modality == 'text' and embeddings is not None:
            msg = 'Please dont specify embeddings for updating a text project'
            raise ValueError(msg)

        if embeddings is not None and len(data) != embeddings.shape[0]:
            msg = 'Expected data and embeddings to be the same length but found lengths {} and {} respectively.'.format()
            raise ValueError(msg)


        # Add new data
        logger.info("Uploading data to Nomic's neural database Atlas.")
        with tqdm(total=len(data) // shard_size) as pbar:
            for i in range(0, len(data), MAX_MEMORY_CHUNK):
                if self.modality == 'embedding':
                    self.add_embeddings(
                        embeddings=embeddings[i: i + MAX_MEMORY_CHUNK, :],
                        data=data[i: i + MAX_MEMORY_CHUNK],
                        shard_size=shard_size,
                        num_workers=num_workers,
                        pbar=pbar,
                    )
                else:
                    self.add_text(
                        data=data[i: i + MAX_MEMORY_CHUNK],
                        shard_size=shard_size,
                        num_workers=num_workers,
                        pbar=pbar,
                    )
        logger.info("Upload succeeded.")

        #Update maps
        # finally, update all the indices
        return self.rebuild_maps()

    def rebuild_maps(self, rebuild_topic_models: bool = False):
        '''
        Rebuilds all maps in a project with the latest state project data state. Maps will not be rebuilt to
        reflect the additions, deletions or updates you have made to your data until this method is called.

        Args:
            rebuild_topic_models: (Default False) - If true, will create new topic models when updating these indices
        '''

        response = requests.post(
            self.atlas_api_path + "/v1/project/update_indices",
            headers=self.header,
            json={
                'project_id': self.id,
                'rebuild_topic_models': rebuild_topic_models
            },
        )

        logger.info(f"Updating maps in project `{self.name}`")
