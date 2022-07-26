"""
The Official Nomic Python Client for Atlas

This class allows for programmatic interactions with Atlas. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize your data.
"""
import concurrent.futures
import json
import os
from typing import Dict, List

import numpy as np
import requests

from .cli import get_api_token


class AtlasClient:
    """The Atlas Client"""

    def __init__(self, hostname: str = 'staging-api-atlas.nomic.ai'):
        '''
        Initializes the Atlas client.

        Args:
            hostname: the hostname where the Atlas back-end is running.
            port: the port where the Atlsa back-end is running.

        '''
        self.atlas_api_path = f"https://{hostname}"
        self.token = str(get_api_token()).strip()
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
            True on success

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
        return True

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


    def _get_project_by_name(self, project_name):
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




    def add_embeddings(self, project_name: str, embeddings: np.array, data: List[Dict]):
        '''
        Adds embeddings to an embedding project. Pair each embedding with meta-data to explore your embeddings.

        Args:
            project_name: The name of the project.
            embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
            data: An [N,] element list of dictionaries containing metadata for each embedding.

        Returns:
            True on success.
        '''

        target_project_id = self._get_project_by_name(project_name=project_name)['id']

        shard_size = 1000

        def send_request(i):
            data_shard = data[i : i + shard_size]
            embedding_shard = embeddings[i : i + shard_size, :].tolist()
            response = requests.post(
                self.atlas_api_path + "/v1/project/data/add/embedding/initial",
                headers=self.header,
                json={'project_id': target_project_id, 'embeddings': embedding_shard, 'data': data_shard},
            )
            return response

        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(send_request, i): i for i in range(0, len(data), shard_size)}
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                if response.status_code != 200:
                    failed.append(futures[future])

        if failed:
            raise ValueError("Failed to upload a subset of datums")

        return True

    def create_index(self, project_name, index_name):

        project = self._get_project_by_name(project_name=project_name)

        index_build_template = {
            'project_id': project['id'],
            'index_name': index_name,
            'indexed_field': None,
            'atomizer_strategies': None,
            'model': None,
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

        if not projection_id:
            print("Could not find a projection being built for this index.")
        else:
            print(f"https://atlas.nomic.ai/scatter/{project['id']}/{projection_id}")


    def add_text(self, project_name: str, data: List[Dict]):
        '''
        Adds text to an Atlas text project. Each text datum consists of a keyed dictionary.
        Args:
            project_name: The name of the project.
            data: A [N,] element list of dictionaries containing your datums.

        Returns:
            True if success.

        '''
        raise NotImplementedError("Building indices for text based projects is not yet implemented in this client.")

    def map_embeddings(self, embeddings: np.array, data: List[Dict]):
        '''
        Generates a map of the given embeddings.

        Args:
            embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
            data: An [N,] element list of dictionaries containing metadata for each embedding.

        Returns:

        '''



        pass

    # def get_projects(self) -> List:
    #     '''
    #     Retrieves all projects available in Atlas.
    #
    #     Returns: A list of accessible projects.
    #     '''
    #     response = requests.get(f"{self.atlas_backend_path}/v1/project/titles")
    #
    #     return response.json()['titles']

    # def get_documents_by_tags(self, project: str) -> Dict:
    #     '''
    #     Retrieves all tagged documents in an Atlas project.
    #
    #     Args:
    #         project: the name of the Atlas project.
    #
    #     Returns: a dictionary grouping document ids by tag name.
    #     '''
    #     response = requests.post(
    #         f"{self.atlas_backend_path}/v1/project/query_all_tags",
    #         data=json.dumps(
    #             {
    #                 'project': project,
    #             }
    #         ),
    #     )
    #     if response.status_code != 200:
    #         raise AtlasException(f"Failed to retrieve documents grouped by ID's. Reason: {response.text}")
    #
    #     return response.json()['results']
    #
    # def get_documents_by_ids(self, project: str, ids: List[str]) -> List[Dict]:
    #     '''
    #     Retrieves documents by a list of ids.
    #
    #     Args:
    #         project: the Atlas project.
    #         ids: a list of document ids
    #
    #     Returns: A list of document JSONs.
    #
    #     '''
    #     response = requests.post(
    #         f"{self.atlas_backend_path}/v1/project/load_ids",
    #         data=json.dumps(
    #             {
    #                 'project': project,
    #                 'ids': ids
    #             }
    #         ),
    #     )
    #     if response.status_code != 200:
    #         raise AtlasException(f"Failed to retrieve documents. Reason: {response.text}")
    #
    #     return response.json()['data']
    #
    # def tag_documents(self, project: str, ids: List[str], tags: List[str]):
    #     """
    #     Adds every tag in `tags` to every document in `ids`
    #
    #     Args:
    #         project: the Atlas project.
    #         ids: A list of document ids to tag.
    #         tags: the set of tags to add.
    #
    #     Returns:
    #         None
    #
    #     """
    #
    #     response = requests.post(
    #         f"{self.atlas_backend_path}/v1/project/tag",
    #         data=json.dumps(
    #             {
    #                 'project': project,
    #                 'ids': ids,
    #                 'tags': tags
    #             }
    #         ),
    #     )
    #     if response.status_code != 200:
    #         raise AtlasException(f"Failed to tag documents. Reason: {response.text}")
    #
    # def untag_documents(self, project: str, ids: List[str], tags: List[str]):
    #     """
    #     Removes every tag in 'tags' from every document in 'ids'.
    #
    #     Args:
    #         project: the Atlas project.
    #         ids: A list of document ids to un-tag.
    #         tags: the set of tags to remove.
    #
    #     Returns:
    #         None
    #
    #     """
    #
    #     response = requests.post(
    #         f"{self.atlas_backend_path}/v1/project/load_ids",
    #         data=json.dumps(
    #             {
    #                 'project': project,
    #                 'ids': ids
    #             }
    #         ),
    #     )
    #
    #     if response.status_code != 200:
    #         raise AtlasException(f"Failed to un-tag documents. Reason: {response.text}")
    #
    #     for id, contents in response.json()['data'].items():
    #         response = requests.post(
    #             f"{self.atlas_backend_path}/v1/project/tag",
    #             data=json.dumps(
    #                 {
    #                     'project': project,
    #                     'ids': [id],
    #                     'tags': [tag for tag in contents['tags'] if tag not in tags]
    #                 }
    #             ),
    #         )
    #         if response.status_code != 200:
    #             raise AtlasException(f"Failed to tag documents. Reason: {response.text}")
    #
    #
    # def get_documents(self, project) -> List[Dict]:
    #     """
    #     Retrieves all documents from a given project.
    #
    #     Args:
    #         project: the Atlas project.
    #
    #     Returns: A list of all documents.
    #
    #     """
    #
    #     response = requests.post(
    #         f"{self.atlas_backend_path}/v1/project/get_all_documents",
    #         data=json.dumps(
    #             {
    #                 'project': project,
    #             }
    #         ),
    #     )
    #     if response.status_code != 200:
    #         raise AtlasException(f"Failed to retrieve documents. Reason: {response.text}")
    #
    #     return response.json()['result']

    # def get_current_selection(self, project: str) -> List[str]:
    #     '''
    #
    #     Retrieves the ids of the document currently selected in the Atlas front-end.
    #
    #     Args:
    #         project: the name of the Atlas project.
    #
    #     Returns: The ids of the currently selected documents.
    #     '''
    #     raise NotImplementedError()
