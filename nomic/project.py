from .atlas import AtlasClass, ATLAS_DEFAULT_ID_FIELD, get_object_size_in_bytes, CreateIndexResponse
from .utils import assert_valid_project_id
from typing import Optional, Union
from loguru import logger
from pyarrow import compute as pc
from typing import List, Dict, Optional, Union
import pyarrow as pa
import requests
import nomic
import base64
import numpy as np
import io
import time

import concurrent
import concurrent.futures
from tqdm import tqdm
import json

DEFAULT_PROJECTION_N_NEIGHBORS = 15
DEFAULT_PROJECTION_EPOCHS = 50
DEFAULT_PROJECTION_SPREAD = 1.0


class AtlasIndex(AtlasClass):
    """
    An AtlasIndex represents a single view of an Atlas Project at a point in time.

    An AtlasIndex typically contains one or more *projections* which are 2d representations of
    the points in the index that you can browse online.
    """

    pass


class AtlasProject(AtlasClass):
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        unique_id_field: Optional[str] = None,
        modality: Optional[str] = None,
        organization_name: Optional[str] = None,
        is_public: bool = True,
        project_id=None,
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
        * **project_id** - An alternative way to retrieve a project is by passing the project_id directly. This only works if a project exissts.
        **Returns:** project_id on success.

        """
        assert name is not None or project_id is not None, "You must pass a name or project_id"

        super().__init__()

        if project_id is not None:
            self.meta = self._get_project_by_id(project_id)
            self.name = self.meta['project_name']
            return

        self.name = name
        try:
            proj = self.get_project(self.name)
            self.meta = self._get_project_by_id(proj['id'])

        except Exception as e:
            if "Could not find project" in str(e):
                assert description is not None, "You must provide a description when creating a new project."
                assert modality is not None, "You must provide a modality when creating a new project."
                logger.info(f"Creating project: {self.name}")
                if unique_id_field is None:
                    unique_id_field = ATLAS_DEFAULT_ID_FIELD
                self.create_project(
                    self.name,
                    description=description,
                    unique_id_field=unique_id_field,
                    modality=modality,
                    organization_name=organization_name,
                    is_public=is_public,
                    reset_project_if_exists=False,
                )
                proj = self.get_project(self.name)
                self.meta = self._get_project_by_id(proj['id'])
            else:
                raise

    def delete(self):
        self.delete_project(project_id=self.id)

    @staticmethod
    def create_project(
        self,
        project_name: str,
        description: str,
        unique_id_field: str,
        modality: str,
        organization_name: Optional[str] = None,
        is_public: bool = True,
        reset_project_if_exists: bool = False,
        add_datums_if_exists: bool = False,
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
        * **reset_project_if_exists** - If the requested project exists in your organization, will delete it and re-create it.
        * **add_datums_if_exists** - Add datums if the project already exists

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

        # check if this project already exists.
        response = requests.post(
            self.atlas_api_path + "/v1/project/search/name",
            headers=self.header,
            json={'organization_name': organization_name, 'project_name': project_name},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to create project: {response.json()}")
        search_results = response.json()['results']

        if search_results:
            existing_project = search_results[0]
            existing_project_id = existing_project['id']
            if reset_project_if_exists:
                logger.info(
                    f"Found existing project `{project_name}` in organization `{organization_name}`. Clearing it of data by request."
                )
                self.delete_project(project_id=existing_project_id)
            else:
                if add_datums_if_exists:
                    logger.info(
                        f"Found existing project `{project_name}` in organization `{organization_name}`. Adding data to this project instead of creating a new one."
                    )
                    return existing_project_id
                else:
                    raise ValueError(
                        f"Project already exists with the name `{project_name}` in organization `{organization_name}`."
                        f"You can add datums to it by settings `add_datums_if_exists = True` or reset it by specifying `reset_project_if_exist=True` on a new upload."
                    )

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

    def project_info(self):
        response = requests.get(
            self.atlas_api_path + f"/v1/project/{self.id}",
            headers=self.header,
        )
        return response.json()

    @property
    def indices(self):
        return self.project_info()['atlas_indices']

    @property
    def projections(self):
        vs = []
        for index in self.indices:
            for projection in index['projections']:
                vs.append(nomic.AtlasProjection(self, projection['id']))
        return vs

    @property
    def id(self):
        return self.meta['id']

    def create_index(
        self,
        index_name: str,
        indexed_field=None,
        colorable_fields: list = [],
        multilingual: bool = False,
        build_topic_model: bool = False,
        projection_n_neighbors=DEFAULT_PROJECTION_N_NEIGHBORS,
        projection_epochs=DEFAULT_PROJECTION_EPOCHS,
        projection_spread=DEFAULT_PROJECTION_SPREAD,
        topic_label_field=None,
        reuse_lm_from=None,
    ) -> CreateIndexResponse:
        '''
        Creates an index in the specified project

        **Parameters:**

        * **project_id** - The ID of the project this index is being built under.
        * **index_name** - The name of the index
        * **indexed_field** - Default None. For text projects, name the data field corresponding to the text to be mapped.
        * **colorable_fields** - The project fields you want to be able to color by on the map. Must be a subset of the projects fields.
        * **multilingual** - Should the map take language into account? If true, points from different languages but semantically similar text are close together.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.
        * **topic_label_field** - A text field to estimate topic labels from.
        * **reuse_lm_from** - An optional index id from the same project whose atoms and embeddings should be reused. Text projects only.

        **Returns:** A link to your map.
        '''
        project = self.meta

        if project['modality'] == 'embedding':
            build_template = {
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
                    {'n_neighbors': projection_n_neighbors, 'n_epochs': projection_epochs, 'spread': projection_spread}
                ),
                'topic_model_hyperparameters': json.dumps(
                    {'build_topic_model': build_topic_model, 'community_description_target_field': topic_label_field}
                ),
            }

        elif project['modality'] == 'text':
            if indexed_field is None:
                raise Exception("You did not specify a field to index. Specify an 'indexed_field'.")

            if indexed_field not in project['project_fields']:
                raise Exception(f"Your index field is not valid. Valid options are: {project['project_fields']}")

            model = 'NomicEmbed'
            if multilingual:
                model = 'NomicEmbedMultilingual'

            build_template = {
                'project_id': project['id'],
                'index_name': index_name,
                'indexed_field': indexed_field,
                'atomizer_strategies': ['document', 'charchunk'],
                'model': model,
                'colorable_fields': colorable_fields,
                'reuse_atoms_and_embeddings_from': reuse_lm_from,
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
            logger.info('Additional info: {}'.format(response.json()))
            raise Exception(response.json['detail'])

        job_id = response.json()['job_id']

        job = requests.get(
            self.atlas_api_path + f"/v1/project/index/job/{job_id}",
            headers=self.header,
        ).json()

        index_id = job['index_id']

        def get_projection_id(project):
            project = requests.get(
                self.atlas_api_path + f"/v1/project/{project['id']}",
                headers=self.header,
            ).json()

            projection_id = None
            for index in self.indices:
                if index['id'] == index_id:
                    projection_id = index['projections'][0]['id']
                    break
            return projection_id

        projection_id = get_projection_id(project)

        if not projection_id:
            time.sleep(5)
            projection_id = get_projection_id(project)

        to_return = {'job_id': job_id, 'index_id': index_id}
        if not projection_id:
            logger.warning(
                "Could not find a map being built for this project. See atlas.nomic.ai/dashboard for map status."
            )
        else:
            if self.credentials['tenant'] == 'staging':
                to_return['map'] = f"https://staging-atlas.nomic.ai/map/{project['id']}/{projection_id}"
            else:
                to_return['map'] = f"https://atlas.nomic.ai/map/{project['id']}/{projection_id}"
            logger.info(f"Created map `{index_name}` in project `{project['project_name']}`: {to_return['map']}")
        to_return['project_id'] = project['id']
        to_return['project_name'] = project['project_name']
        return CreateIndexResponse(**to_return)

    def __repr__(self):
        m = self.meta
        return f"Nomic project: <{m}>"

    def add_text(
        self,
        data: List[Dict],
        shard_size=1000,
        num_workers=10,
        replace_empty_string_values_with_string_null=True,
        pbar=None,
    ):
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
        project_id = self.id

        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        # Check if this is a progressive project
        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project_id}",
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
            self._validate_and_correct_user_supplied_metadata(
                data=data,
                project=project,
                replace_empty_string_values_with_string_null=replace_empty_string_values_with_string_null,
            )
        except BaseException as e:
            raise e

        upload_endpoint = "/v1/project/data/add/json/initial"
        if progressive:
            upload_endpoint = "/v1/project/data/add/json/progressive"

        # Actually do the upload
        def send_request(i):
            data_shard = data[i : i + shard_size]
            if get_object_size_in_bytes(data_shard) > 8000000:
                raise Exception(
                    "Your metadata upload shards are to large. Try decreasing the shard size or removing un-needed fields from the metadata."
                )
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
            logger.info("Uploading text to Atlas.")
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
                            logger.error(f"Shard upload failed: {response.json()}")
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
                                    f"Connection failed for records {start_point}-{start_point + shard_size}, retrying."
                                )
                                failure_fraction = errors_504 / (failed + succeeded + errors_504)
                                if failure_fraction > 0.25 and errors_504 > shard_size * 3:
                                    raise RuntimeError(
                                        "Atlas is under high load and cannot ingest datums at this time. Please try again later."
                                    )
                                new_submission = executor.submit(send_request, start_point)
                                futures[new_submission] = start_point
                                response.close()
                            else:
                                logger.error(f"Shard upload failed: {response}")
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
            logger.warning(f"Failed to upload {len(failed) * shard_size} datums")
        if close_pbar:
            if failed:
                logger.warning("Text upload partially succeeded.")
            else:
                logger.info("Text upload succeeded.")

        return True

    def add_embeddings(
        self,
        embeddings: np.array,
        data: List[Dict],
        shard_size=1000,
        num_workers=10,
        replace_empty_string_values_with_string_null=True,
        pbar=None,
    ):
        '''
        Adds embeddings to an embedding project. Pair each embedding with meta-data to explore your embeddings.

        **Parameters:**

        * **embeddings** - An [N,d] numpy array containing the batch of N embeddings to add.
        * **data** - An [N,] element list of dictionaries containing metadata for each embedding.
        * **shard_size** - Embeddings are uploaded in parallel by many threads. Adjust the number of embeddings to upload by each worker.
        * **num_workers** - The number of worker threads to upload embeddings with.
        * **replace_empty_string_values_with_string_null** - Replaces empty values in metadata with null. If false, will fail if empty values are supplied.

        **Returns:** True on success.
        '''
        project_id = self.id
        assert_valid_project_id(project_id)

        # Each worker currently is to slow beyond a shard_size of 5000
        shard_size = min(shard_size, 5000)

        # Check if this is a progressive project
        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project_id}",
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
            self._validate_and_correct_user_supplied_metadata(
                data=data,
                project=project,
                replace_empty_string_values_with_string_null=replace_empty_string_values_with_string_null,
            )
        except BaseException as e:
            raise e

        upload_endpoint = "/v1/project/data/add/embedding/initial"
        if progressive:
            upload_endpoint = "/v1/project/data/add/embedding/progressive"

        # Actually do the upload
        def send_request(i):
            data_shard = data[i : i + shard_size]

            if get_object_size_in_bytes(data_shard) > 8000000:
                raise Exception(
                    "Your metadata upload shards are to large. Try decreasing the shard size or removing un-needed fields from the metadata."
                )
            embedding_shard = embeddings[i : i + shard_size, :]

            bytesio = io.BytesIO()
            np.save(bytesio, embedding_shard)
            response = requests.post(
                self.atlas_api_path + upload_endpoint,
                headers=self.header,
                json={
                    'project_id': project_id,
                    'embeddings': base64.b64encode(bytesio.getvalue()).decode('utf-8'),
                    'data': data_shard,
                },
            )
            return response

        # if this method is being called internally, we pass a global progress bar
        close_pbar = False
        if pbar is None:
            logger.info("Uploading embeddings to Atlas.")
            close_pbar = True
            pbar = tqdm(total=int(embeddings.shape[0]) // shard_size)
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
                            logger.error(f"Shard upload failed: {response.json()}")
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
                                    f"Connection failed for records {start_point}-{start_point + shard_size}, retrying."
                                )
                                failure_fraction = errors_504 / (failed + succeeded + errors_504)
                                if failure_fraction > 0.25 and errors_504 > shard_size * 3:
                                    raise RuntimeError(
                                        "Atlas is under high load and cannot ingest datums at this time. Please try again later."
                                    )
                                new_submission = executor.submit(send_request, start_point)
                                futures[new_submission] = start_point
                                response.close()
                            else:
                                logger.error(f"Shard upload failed: {response}")
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
                logger.warning("Embedding upload partially succeeded.")
            else:
                logger.info("Embedding upload succeeded.")

        return True

    # def upload_embeddings(self, table: pa.Table, embedding_column: str = '_embedding'):
    #     """
    #     Uploads a single Arrow table to Atlas.
    #     """
    #     dimensions = table[embedding_column].type.list_size
    #     embs = pc.list_flatten(table[embedding_column]).to_numpy().reshape(-1, dimensions)
    #     self.atlas_client.add_embeddings(
    #         project_id=self.id, embeddings=embs, data=table.drop([embedding_column]).to_pylist(), shard_size=1500
    #     )
