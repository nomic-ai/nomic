"""
The Official Nomic Python Client for Atlas

This class allows for programmatic interactions with Atlas. Initialize AtlasClient in any Python context such as a script
or in a Jupyter Notebook in order to retrieve documents, annotations and tags made in the Atlas front-end application.
"""
import json
from typing import Dict, List

import requests

from atlas_client.exceptions import AtlasException
from atlas_client.pipeline import Pipeline


class AtlasClient:
    """The Atlas Client"""

    def __init__(self, hostname: str = 'localhost', port: str = '80'):
        '''
        Initializes the Atlas client.

        Args:
            hostname: the hostname where the Atlas back-end is running.
            port: the port where the Atlsa back-end is running.

        '''
        self.atlas_api_path = f"https://{hostname}:{port}"

    # def create_project(self, project: str, data: List[Dict], id_field: str = '_id') -> None:
    #     '''
    #     Creates a project in Atlas.
    #
    #     Args:
    #         project: the name of the Atlas project.
    #         data: The list of data to upload.
    #         id_field: The field representing a documents unique identifier.
    #
    #
    #     '''
    #     raise NotImplementedError() #not yet tested
    #     if id_field not in data[0]:
    #         raise ValueError(f"Your data does not contain the unique id field `{id_field}`")
    #     response = requests.post(
    #         f"{self.atlas_backend_path}/v1/project/create",
    #         data=json.dumps(
    #             {
    #                 'project': project,
    #                 'id_field': id_field,
    #                 'data': data,
    #             }
    #         ),
    #     )
    #     if response.status_code != 200:
    #         raise AtlasException(f"Project creation failed. Reason: {response.text}")
    #
    # def create_index(self, project: str, pipeline: Pipeline) -> None:
    #     '''
    #     Add an index to the specified Atlas project.
    #
    #     Args:
    #         project: the name of the project the index is being added to.
    #         pipeline: A configured Pipeline object defining the parameters of each step of the pipeline.
    #
    #     '''
    #     raise NotImplementedError()

    def get_projects(self) -> List:
        '''
        Retrieves all projects available in Atlas.

        Returns: A list of accessible projects.
        '''
        response = requests.get(
            f"{self.atlas_backend_path}/v1/project/titles"
        )

        return response.json()['titles']

    def get_documents_by_tags(self, project: str) -> Dict:
        '''
        Retrieves all tagged documents in an Atlas project.

        Args:
            project: the name of the Atlas project.

        Returns: a dictionary grouping document ids by tag name.
        '''
        response = requests.post(
            f"{self.atlas_backend_path}/v1/project/query_all_tags",
            data=json.dumps(
                {
                    'project': project,
                }
            ),
        )
        if response.status_code != 200:
            raise AtlasException(f"Failed to retrieve documents grouped by ID's. Reason: {response.text}")

        return response.json()['results']

    def get_documents_by_ids(self, project: str, ids: List[str]) -> List[Dict]:
        '''
        Retrieves documents by a list of ids.

        Args:
            project: the Atlas project.
            ids: a list of document ids

        Returns: A list of document JSONs.

        '''
        response = requests.post(
            f"{self.atlas_backend_path}/v1/project/load_ids",
            data=json.dumps(
                {
                    'project': project,
                    'ids': ids
                }
            ),
        )
        if response.status_code != 200:
            raise AtlasException(f"Failed to retrieve documents. Reason: {response.text}")

        return response.json()['data']

    def tag_documents(self, project: str, ids: List[str], tags: List[str]):
        """
        Adds every tag in `tags` to every document in `ids`

        Args:
            project: the Atlas project.
            ids: A list of document ids to tag.
            tags: the set of tags to add.

        Returns:
            None

        """

        response = requests.post(
            f"{self.atlas_backend_path}/v1/project/tag",
            data=json.dumps(
                {
                    'project': project,
                    'ids': ids,
                    'tags': tags
                }
            ),
        )
        if response.status_code != 200:
            raise AtlasException(f"Failed to tag documents. Reason: {response.text}")

    def untag_documents(self, project: str, ids: List[str], tags: List[str]):
        """
        Removes every tag in 'tags' from every document in 'ids'.

        Args:
            project: the Atlas project.
            ids: A list of document ids to un-tag.
            tags: the set of tags to remove.

        Returns:
            None

        """

        response = requests.post(
            f"{self.atlas_backend_path}/v1/project/load_ids",
            data=json.dumps(
                {
                    'project': project,
                    'ids': ids
                }
            ),
        )

        if response.status_code != 200:
            raise AtlasException(f"Failed to un-tag documents. Reason: {response.text}")

        for id, contents in response.json()['data'].items():
            response = requests.post(
                f"{self.atlas_backend_path}/v1/project/tag",
                data=json.dumps(
                    {
                        'project': project,
                        'ids': [id],
                        'tags': [tag for tag in contents['tags'] if tag not in tags]
                    }
                ),
            )
            if response.status_code != 200:
                raise AtlasException(f"Failed to tag documents. Reason: {response.text}")


    def get_documents(self, project) -> List[Dict]:
        """
        Retrieves all documents from a given project.

        Args:
            project: the Atlas project.

        Returns: A list of all documents.

        """

        response = requests.post(
            f"{self.atlas_backend_path}/v1/project/get_all_documents",
            data=json.dumps(
                {
                    'project': project,
                }
            ),
        )
        if response.status_code != 200:
            raise AtlasException(f"Failed to retrieve documents. Reason: {response.text}")

        return response.json()['result']

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
