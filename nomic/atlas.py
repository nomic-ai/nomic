"""
This class allows for programmatic interactions with Atlas - Nomics neural database. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize and interact with your unstructured data.
"""
import concurrent.futures
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pydantic
import requests
from pydantic import BaseModel, Field
from tqdm import tqdm
from wonderwords import RandomWord

from .cli import get_api_credentials


class CreateIndexResponse(BaseModel):
    map: Optional[str] = Field(
        None, description="A link to the map this index creates. May take some time to be ready so check the job state."
    )
    job_id: str = Field(..., description="The job_id to track the progress of the index build.")
    index_id: str = Field(..., description="The unique identifier of the index being built.")


class AtlasClient:
    """The Atlas Client"""

    def __init__(self):
        '''
        Initializes the Atlas client.

        '''

        credentials = get_api_credentials()

        if credentials['tenant'] == 'staging':
            hostname = 'staging-api-atlas.nomic.ai'
        elif credentials['tenant'] == 'production':
            hostname = 'api-atlas.nomic.ai'
        else:
            raise ValueError("Invalid tenant.")

        self.atlas_api_path = f"https://{hostname}"
        self.token = credentials['token']
        self.header = {"Authorization": f"Bearer {self.token}"}

        if self.token:
            response = requests.get(
                self.atlas_api_path + "/v1/user",
                headers=self.header,
            )
            if not response.status_code == 200:
                print("Your authorization token is no longer valid.")
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
            raise ValueError("Your authorization token is no longer valid.")

        return response.json()

    def create_project(
        self, project_name: str, description: str, unique_id_field: str, modality: str, is_public: bool = True
    ):
        '''
        Creates an Atlas project. Atlas projects store data (text, embeddings, etc) that you can organize by building indices.

        Args:
            project_name: The name of the project.
            description: A description for the project.
            unique_id_field: The field that uniquely identifies each datum. If a datum does not contain this field, it will be added and assigned a random unique ID.
            modality: The data modality of this project. Currently, Atlas supports either `text` or `embedding` modality projects.
            is_public: Should this project be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.

        Returns:
            project_id on success.

        '''

        organization_id = self._get_current_users_main_organization()

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

        Returns:
            The ID of the current users default organization.

        '''

        user = self._get_current_user()
        for organization in user['organizations']:
            if organization['user_id'] == user['sub'] and organization['access_role'] == 'OWNER':
                return organization['organization_id']

    def _get_project_by_name(self, project_name: str):
        '''
        Retrieves a project that a user owns by name.
        Args:
            project_name: the project name

        Returns:

        '''

        organization_id = self._get_current_users_main_organization()

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

    def add_embeddings(self, project_id: str, embeddings: np.array, data: List[Dict], shard_size=1000, num_workers=10):
        '''
        Adds embeddings to an embedding project. Pair each embedding with meta-data to explore your embeddings.

        Args:
            project_id: The id of the project you are adding emebeddings to.
            embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
            data: An [N,] element list of dictionaries containing metadata for each embedding.
            shard_size: Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
            num_workers: The number of worker threads to upload embeddings with.

        Returns:
            True on success.
        '''

        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        def send_request(i):
            data_shard = data[i : i + shard_size]
            embedding_shard = embeddings[i : i + shard_size, :].tolist()
            response = requests.post(
                self.atlas_api_path + "/v1/project/data/add/embedding/initial",
                headers=self.header,
                json={'project_id': project_id, 'embeddings': embedding_shard, 'data': data_shard},
            )
            return response

        failed = []

        print("Uploading embeddings to Nomic.")
        with tqdm(total=int(embeddings.shape[0]) // shard_size) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(send_request, i): i for i in range(0, len(data), shard_size)}
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    pbar.update(1)
                    if response.status_code != 200:
                        failed.append(futures[future])

        if failed:
            raise ValueError("Failed to upload a subset of datums")
        print("Embedding upload succeeded.")

        return True

    def create_index(self, project_id: str, index_name: str, colorable_fields=[]):
        '''
        Creates an index in the specified project
        Args:
            project_id: The project id to build the index for.
            index_name: The name of the index
            colorable_fields: The project fields you want to be able to color by on the map. Must be a subset of the projects fields.

        Returns:
            CreateIndexResponse



        '''

        project = self._get_project_by_id(project_id=project_id)

        index_build_template = {
            'project_id': project['id'],
            'index_name': index_name,
            'indexed_field': None,
            'atomizer_strategies': None,
            'model': None,
            'colorable_fields': colorable_fields,
            'model_hyperparameters': None,
            'nearest_neighbor_index': 'HNSWIndex',
            'nearest_neighbor_index_hyperparameters': json.dumps({'space': 'l2', 'ef_construction': 100, 'M': 16}),
            'projection': 'UMAPProjection',
            'projection_hyperparameters': json.dumps(
                {'n_neighbors': 15, 'min_dist': 3e-2, 'force_approximation_algorithm': True}
            ),
        }

        if project['modality'] == 'embedding':
            response = requests.post(
                self.atlas_api_path + "/v1/project/index/create",
                headers=self.header,
                json=index_build_template,
            )
            job_id = response.json()['job_id']

        elif project['modality'] == 'text':
            raise NotImplementedError("Building indices for text based projects is not yet implemented in this client.")

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
            print("Could not find a projection being built for this index.")
        else:
            to_return['map'] = f"https://atlas.nomic.ai/map/{project['id']}/{projection_id}"

        return CreateIndexResponse(**to_return)

    def add_text(self, project_id: str, data: List[Dict]):
        '''
        Adds text to an Atlas text project. Each text datum consists of a keyed dictionary.
        Args:
            project_id: The id of the project to add text to.
            data: A [N,] element list of dictionaries containing your datums.

        Returns:
            True if success.

        '''
        raise NotImplementedError("Building indices for text based projects is not yet implemented in this client.")

    def map_embeddings(self, embeddings: np.array, data: List[Dict], id_field='id', is_public=True, colorable_fields=[], num_workers=10):
        '''
        Generates a map of the given embeddings.

        Args:
            embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
            data: An [N,] element list of dictionaries containing metadata for each embedding.
            id_field: Each datums unique id field.
            colorable_fields: The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
            is_public: Should this embedding map be public or require organizational sign-in to view?
            num_workers: number of workers to use when sending data.

        Returns:
            CreateIndexResponse

        '''

        def get_random_name():
            random_words = RandomWord()
            return f"{random_words.word(include_parts_of_speech=['nouns'])}-{random_words.word(include_parts_of_speech=['adjectives'])}"

        project_name = get_random_name()
        index_name = get_random_name()

        project_id = self.create_project(
            project_name=project_name,
            description=project_name,
            unique_id_field=id_field,
            modality='embedding',
            is_public=is_public,
        )

        shard_size = 1000
        if embeddings.shape[0] > 10000:
            shard_size = 5000

        self.add_embeddings(
            project_id=project_id, embeddings=embeddings, data=data, shard_size=shard_size, num_workers=num_workers
        )

        response = self.create_index(project_id=project_id, index_name=index_name, colorable_fields=colorable_fields)

        return response
