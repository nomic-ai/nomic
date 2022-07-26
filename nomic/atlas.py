"""
The Official Nomic Python Client for Atlas

This class allows for programmatic interactions with Atlas. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook to organize your data.
"""
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

    def _get_user(self):
        response = requests.get(
            self.atlas_api_path + "/v1/user",
            headers=self.header,
        )
        if not response.status_code == 200:
            print("Your authorization token is no longer valid.")
            exit()
        return response.json()

    def create_project(self, project_name: str, description: str, unique_id_field: str, modality: str, is_public: bool = True):
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

        user = self._get_user()
        if len(user['organizations']) > 1:
            raise NotImplementedError("This client does not support users in more than one organization yet.")

        organization_id = user['organizations'][0]['organization_id']

        if modality not in ['text', 'embedding']:
            raise ValueError("Atlas currently only supports 'text' and 'embedding' projects.")

        response = requests.post(
            self.atlas_api_path + "/v1/project/create",
            headers=self.header,
            json={
                'organization_id': organization_id,
                'project_name': project_name,
                'description': description,
                'unique_id_field': unique_id_field,
                'modality': modality,
                'is_public': is_public
            },
        )
        if response.status_code != 201:
            raise Exception(f"Failed to create project: {response.json()}")
        return True


    def add_embeddings(self,
                       project: str,
                       embeddings: np.array,
                       data: List[Dict]
                       ):
        '''
        Adds embeddings to an embedding project. Pair each embedding with meta-data to explore your embeddings.

        Args:
            project: The name of the project.
            embeddings: An [N,d] numpy array containing the batch of N embeddings to add.
            data: An [N,] element list of dictionaries containing metadata for each embedding.

        Returns:
            True on success.
        '''
        pass


    def add_text(self, project: str, data: List[Dict]):
        '''
        Adds text to an Atlas text project. Each text datum consists of a keyed dictionary.
        Args:
            project: The name of the project.
            data: A [N,] element list of dictionaries containing your datums.

        Returns:
            True if success.

        '''
        pass

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
