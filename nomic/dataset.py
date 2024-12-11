import base64
import concurrent
import concurrent.futures
import importlib.metadata
import io
import json
import os
import re
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from loguru import logger
from pandas import DataFrame
from PIL import Image
from pyarrow import compute as pc
from pyarrow import feather, ipc
from tqdm import tqdm

from .cli import refresh_bearer_token, validate_api_http_response
from .data_inference import (
    NomicDuplicatesOptions,
    NomicEmbedOptions,
    NomicProjectOptions,
    NomicTopicOptions,
    convert_pyarrow_schema_for_atlas,
)
from .data_operations import AtlasMapData, AtlasMapDuplicates, AtlasMapEmbeddings, AtlasMapTags, AtlasMapTopics
from .settings import *
from .utils import assert_valid_project_id, download_feather


class AtlasUser:
    def __init__(self):
        self.credentials = refresh_bearer_token()


class AtlasClass(object):
    def __init__(self):
        """
        Initializes the Atlas client.
        """

        if self.credentials["tenant"] == "staging":
            api_hostname = "staging-api-atlas.nomic.ai"
            web_hostname = "staging-atlas.nomic.ai"
        elif self.credentials["tenant"] == "production":
            api_hostname = "api-atlas.nomic.ai"
            web_hostname = "atlas.nomic.ai"
        elif self.credentials["tenant"] == "enterprise":
            api_hostname = self.credentials["api_domain"]
            web_hostname = self.credentials["frontend_domain"]
        else:
            raise ValueError("Invalid tenant.")

        self.atlas_api_path = f"https://{api_hostname}"
        self.web_path = f"https://{web_hostname}"

        try:
            override_api_path = os.environ["ATLAS_API_PATH"]
        except KeyError:
            override_api_path = None

        if override_api_path:
            self.atlas_api_path = override_api_path

        token = self.credentials["token"]
        self.token = token

        try:
            version = importlib.metadata.version("nomic")
        except Exception:
            version = "unknown"

        self.header = {"Authorization": f"Bearer {token}", "User-Agent": f"py-nomic/{version}"}

        if self.token:
            response = requests.get(
                self.atlas_api_path + "/v1/user",
                headers=self.header,
            )
            if "X-AtlasWarning" in response.headers:
                logger.warning(response.headers["X-AtlasWarning"])
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

        response = requests.get(
            api_base_path + "/v1/user",
            headers=self.header,
        )
        response = validate_api_http_response(response)
        if not response.status_code == 200:
            raise ValueError("Your authorization token is no longer valid. Run `nomic login` to obtain a new one.")

        return response.json()

    def _validate_map_data_inputs(self, colorable_fields, id_field, data_sample):
        """Validates inputs to map data calls."""

        if not isinstance(colorable_fields, list):
            raise ValueError("colorable_fields must be a list of fields")

        if id_field in colorable_fields:
            raise Exception(f"Cannot color by unique id field: {id_field}")

        for field in colorable_fields:
            if field not in data_sample:
                raise Exception(f"Cannot color by field `{field}` as it is not present in the metadata.")

    def _get_current_users_main_organization(self):
        """
        Retrieves the ID of the current users default organization.

        **Returns:** The ID of the current users default organization

        """

        user = self._get_current_user()

        for organization in user["organizations"]:
            if organization.get("plan_type") and organization.get("plan_type") == "enterprise":
                return organization

        if user["default_organization"]:
            for organization in user["organizations"]:
                if organization["organization_id"] == user["default_organization"]:
                    return organization

        for organization in user["organizations"]:
            if organization["user_id"] == user["sub"] and organization["access_role"] == "OWNER":
                return organization

        for organization in user["organizations"]:
            if organization["user_id"] == user["sub"]:
                return organization
        return {}

    def _delete_project_by_id(self, project_id):
        response = requests.post(
            self.atlas_api_path + "/v1/project/remove",
            headers=self.header,
            json={"project_id": project_id},
        )

    def _get_project_by_id(self, project_id: str):
        """

        Args:
            project_id: The project id

        Returns:
            Returns the requested dataset.
        """

        assert_valid_project_id(project_id)

        response = requests.get(
            self.atlas_api_path + f"/v1/project/{project_id}",
            headers=self.header,
        )

        if response.status_code != 200:
            raise Exception(f"Could not access dataset with id {project_id}: {response.text}")

        return response.json()

    def _get_organization_by_slug(self, slug: str):
        """

        Args:
            slug: The organization slug

        Returns:
            An organization id
        """

        if "/" in slug:
            slug = slug.split("/")[0]

        response = requests.get(
            self.atlas_api_path + f"/v1/organization/{slug}",
            headers=self.header,
        )
        if response.status_code != 200:
            raise Exception(f"Organization not found: {slug}")

        return response.json()["id"]

    def _get_dataset_by_slug_identifier(self, identifier: str):
        """

        Args:
            identifier: the organization slug and dataset slug seperated by a slash

        Returns:
            Returns the requested dataset.
        """

        if not self.is_valid_dataset_identifier(identifier=identifier):
            raise Exception("Invalid dataset identifier")

        organization_slug = identifier.split("/")[0]
        project_slug = identifier.split("/")[1]
        response = requests.get(
            self.atlas_api_path + f"/v1/project/{organization_slug}/{project_slug}",
            headers=self.header,
        )

        if response.status_code == 403:
            raise ValueError(response.json()["detail"])

        if response.status_code != 200:
            return None

        return response.json()

    def is_valid_dataset_identifier(self, identifier: str):
        """
        Checks if a string is a valid identifier for a dataset

        Args:
            identifer: the organization slug and dataset slug separated by a slash

        Returns:
            Returns the requested dataset.
        """
        slugs = identifier.split("/")
        if "/" not in identifier or len(slugs) != 2:
            return False
        return True

    def _get_index_job(self, job_id: str):
        """

        Args:
            job_id: The job id to retrieve the state of.

        Returns:
            Job ID meta-data.
        """

        response = requests.get(
            self.atlas_api_path + f"/v1/project/index/job/{job_id}",
            headers=self.header,
        )

        if response.status_code != 200:
            raise Exception(f"Could not access job state: {response.text}")

        return response.json()

    def _validate_and_correct_arrow_upload(self, data: pa.Table, project: "AtlasDataset") -> pa.Table:
        """
        Private method. validates upload data against the dataset arrow schema, and associated other checks.

        1. If unique_id_field is specified, validates that each datum has that field. If not, adds it and then notifies the user that it was added.

        Args:
            data: an arrow table.
            project: the atlas dataset you are validating the data for.

        Returns:

        """
        if not isinstance(data, pa.Table):
            raise Exception("Invalid data type for upload: {}".format(type(data)))

        if project.meta["modality"] == "text":
            if "_embeddings" in data:
                msg = "Can't add embeddings to a text project."
                raise ValueError(msg)
        if project.meta["modality"] == "embedding":
            if "_embeddings" not in data.column_names:
                msg = "Must include embeddings in embedding dataset upload."
                raise ValueError(msg)

        if project.id_field not in data.column_names:
            raise ValueError(f"Data must contain the ID column `{project.id_field}`")

        seen = set()
        for col in data.column_names:
            if col.lower() in seen:
                raise ValueError(
                    f"Two different fields have the same lowercased name, `{col}`" ": you must use unique column names."
                )
            seen.add(col.lower())

        if project.schema is None:
            project._schema = convert_pyarrow_schema_for_atlas(data.schema)
        # Reformat to match the schema of the project.
        # This includes shuffling the order around if necessary,
        # filling in nulls, etc.
        reformatted = {}

        if data[project.id_field].null_count > 0:
            raise ValueError(
                f"{project.id_field} must not contain null values, but {data[project.id_field].null_count} found."
            )

        assert project.schema is not None, "Project schema not found."

        for field in project.schema:
            if field.name in data.column_names:
                # Allow loss of precision in dates and ints, etc.
                reformatted[field.name] = data[field.name].cast(field.type, safe=False)
            else:
                raise KeyError(
                    f"Field {field.name} present in table schema not found in data. Present fields: {data.column_names}"
                )
            if pa.types.is_string(field.type):
                # Ugly temporary measures
                if data[field.name].null_count > 0:
                    logger.warning(
                        f"Replacing {data[field.name].null_count} null values for field {field.name} with string 'null'. This behavior will change in a future version."
                    )
                    reformatted[field.name] = pc.fill_null(reformatted[field.name], "null")
                if pa.compute.any(pa.compute.equal(pa.compute.binary_length(reformatted[field.name]), 0)):  # type: ignore
                    mask = pa.compute.equal(pa.compute.binary_length(reformatted[field.name]), 0).combine_chunks()  # type: ignore
                    assert pa.types.is_boolean(mask.type)  # type: ignore
                    reformatted[field.name] = pa.compute.replace_with_mask(reformatted[field.name], mask, "null")  # type: ignore
        for field in data.schema:
            if not field.name in reformatted:
                if field.name == "_embeddings":
                    reformatted["_embeddings"] = data["_embeddings"]
                else:
                    logger.warning(f"Field {field.name} present in data, but not found in table schema. Ignoring.")
        data = pa.Table.from_pydict(reformatted, schema=project.schema)

        if project.meta["insert_update_delete_lock"]:
            raise Exception("Project is currently indexing and cannot ingest new datums. Try again later.")

        # The following two conditions should never occur given the above, but just in case...
        assert project.id_field in data.column_names, f"Upload does not contain your specified id_field"

        if not pa.types.is_string(data[project.id_field].type):
            logger.warning(f"id_field is not a string. Converting to string from {data[project.id_field].type}")
            data = data.drop([project.id_field]).append_column(
                project.id_field, data[project.id_field].cast(pa.string())
            )

        for key in data.column_names:
            if key.startswith("_"):
                if key == "_embeddings" or key == "_blob_hash":
                    continue
                raise ValueError("Metadata fields cannot start with _")
        if pa.compute.max(pa.compute.utf8_length(data[project.id_field])).as_py() > 36:  # type: ignore
            first_match = data.filter(
                pa.compute.greater(pa.compute.utf8_length(data[project.id_field]), 36)  # type: ignore
            ).to_pylist()[0][project.id_field]
            raise ValueError(
                f"The id_field {first_match} is greater than 36 characters. Atlas does not support id_fields longer than 36 characters."
            )
        return data

    def _get_organization(self, organization_slug=None, organization_id=None) -> Tuple[str, str]:
        """
        Gets an organization by either its name or id.

        Args:
            organization_slug: the slug of the organization
            organization_id: the id of the organization

        Returns:
            The organization_slug and organization_id if one was found.

        """

        if organization_slug is None:
            if organization_id is None:  # default to current users organization (the one with their name)
                organization = self._get_current_users_main_organization()
                organization_slug = organization["slug"]
                organization_id = organization["organization_id"]
            else:
                raise NotImplementedError("Getting organization by a specific ID is not yet implemented.")

        else:
            try:
                organization_id = self._get_organization_by_slug(slug=organization_slug)
            except Exception:
                user = self._get_current_user()
                users_organizations = [org["slug"] for org in user["organizations"]]
                raise Exception(
                    f"No such organization exists: {organization_slug}. You have access to the following organizations: {users_organizations}"
                )

        return organization_slug, organization_id


class AtlasIndex:
    """
    An AtlasIndex represents a single view of an AtlasDataset at a point in time.

    An AtlasIndex typically contains one or more *projections* which are 2D representations of
    the points in the index that you can browse online.
    """

    def __init__(self, atlas_index_id, name, indexed_field, projections):
        """Initializes an Atlas index. Atlas indices organize data and store views of the data as maps."""
        self.id = atlas_index_id
        self.name = name
        self.indexed_field = indexed_field
        self.projections = projections

    def _repr_html_(self):
        return "<br>".join([d._repr_html_() for d in self.projections])


class AtlasProjection:
    """
    Interact and access state of an Atlas Map including text/vector search.
    This class should not be instantiated directly.

    Instead instantiate an AtlasDataset and use the dataset.maps attribute to retrieve an AtlasProjection.
    """

    def __init__(self, dataset: "AtlasDataset", atlas_index_id: str, projection_id: str, name):
        """
        Creates an AtlasProjection.
        """
        self.dataset = dataset
        self.id = projection_id
        self.atlas_index_id = atlas_index_id
        self.projection_id = projection_id
        self.name = name
        self._duplicates = None
        self._embeddings = None
        self._topics = None
        self._tags = None
        self._tile_data = None
        self._data = None
        self._schema = None
        self._manifest_tb: Optional[pa.Table] = None
        self._columns: List[Tuple[str, str]] = []

    @property
    def map_link(self):
        """
        Retrieves a map link.
        """
        return f"{self.dataset.web_path}/data/{self.dataset.meta['organization_slug']}/{self.dataset.meta['slug']}/map"
        # return f"{self.project.web_path}/data/{self.project.meta['organization_slug']}/{self.project.meta['slug']}/map"

    @property
    def dataset_link(self):
        """
        Retrieves a dataset link.
        """
        return f"{self.dataset.web_path}/data/{self.dataset.meta['organization_slug']}/{self.dataset.meta['slug']}"

    @property
    def _status(self):
        response = requests.get(
            self.dataset.atlas_api_path + f"/v1/project/index/job/progress/{self.atlas_index_id}",
            headers=self.dataset.header,
        )
        if response.status_code != 200:
            raise Exception(response.text)

        content = response.json()
        return content

    def __str__(self):
        return f"{self.name}: {self.dataset_link}"

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
        # Don't make an iframe if the dataset is locked.
        state = self._status["index_build_stage"]
        if state != "Completed":
            return f"""Atlas Projection {self.name}. Status {state}. <a target="_blank" href="{self.map_link}">view online</a>"""
        return f"""
            <h3>Project: {self.dataset.slug}</h3>
            {self._embed_html()}
            """

    @property
    def duplicates(self):
        """Duplicate detection state"""
        if self.dataset.is_locked:
            raise Exception("Dataset is locked! Please wait until the dataset is unlocked to access duplicates.")
        if self._duplicates is None:
            self._duplicates = AtlasMapDuplicates(self)
        return self._duplicates

    @property
    def topics(self):
        """Topic state"""
        if self.dataset.is_locked:
            raise Exception(
                "Dataset is locked for state access! Please wait until the dataset is unlocked to access topics."
            )
        if self._topics is None:
            self._topics = AtlasMapTopics(self)
        return self._topics

    @property
    def embeddings(self):
        """Embedding state"""
        if self.dataset.is_locked:
            raise Exception(
                "Dataset is locked for state access! Please wait until the dataset is unlocked to access embeddings."
            )
        if self._embeddings is None:
            self._embeddings = AtlasMapEmbeddings(self)
        return self._embeddings

    @property
    def tags(self):
        """Tag state"""
        if self.dataset.is_locked:
            raise Exception(
                "Dataset is locked for state access! Please wait until the dataset is unlocked to access tags."
            )
        if self._tags is None:
            self._tags = AtlasMapTags(self)
        return self._tags

    @property
    def data(self):
        """Metadata state"""
        if self.dataset.is_locked:
            raise Exception(
                "Dataset is locked for state access! Please wait until the dataset is unlocked to access data."
            )
        if self._data is None:
            self._data = AtlasMapData(self)
        return self._data

    @property
    def schema(self):
        """Projection arrow schema"""
        if self.dataset.is_locked:
            raise Exception(
                "Dataset is locked for state access! Please wait until the dataset is unlocked to access data."
            )
        if self._schema is None:
            response = requests.get(
                self.dataset.atlas_api_path + f"/v1/project/projection/{self.projection_id}/schema",
                headers=self.dataset.header,
            )
            if response.status_code != 200:
                raise Exception(response.text)

            content = response.content
            self._schema = ipc.read_schema(io.BytesIO(content))
        return self._schema

    @property
    def _registered_columns(self) -> List[Tuple[str, str]]:
        "Returns [(field_name, sidecar_name), ...]"
        if self._columns:
            return self._columns
        self._columns = []
        for field in self.schema:
            sidecar_name = json.loads(field.metadata.get(b"sidecar_name", b'""'))
            if sidecar_name is not None:
                self._columns.append((field.name, sidecar_name))
        return self._columns

    @property
    def _manifest(self) -> pa.Table:
        """
        Returns the tile manifest for the projection.
        Tile manifest is in quadtree order. All quadtree operations should
        depend on tile manifest to ensure consistency.
        """
        if self._manifest_tb is not None:
            return self._manifest_tb

        manifest_path = self.tile_destination / "manifest.feather"
        manifest_url = (
            self.dataset.atlas_api_path
            + f"/v1/project/{self.dataset.id}/index/projection/{self.id}/quadtree/manifest.feather"
        )

        download_feather(manifest_url, manifest_path, headers=self.dataset.header, overwrite=False)
        self._manifest_tb = feather.read_table(manifest_path, memory_map=False)
        return self._manifest_tb

    def _get_sidecar_from_field(self, field: str) -> str:
        """
        Returns the sidecar name for a given field.

        Args:
            field: the name of the field
        """
        for f, sidecar in self._registered_columns:
            if field == f:
                return sidecar
        raise ValueError(f"Field {field} not found in registered columns.")

    def _download_sidecar(self, sidecar_name, overwrite: bool = False) -> List[Path]:
        """
        Downloads sidecar files from the quadtree
        Args:
            sidecar_name: the name of the sidecar file
            overwrite: if True then overwrite existing feather files.

        Returns:
            List of downloaded feather files.
        """
        downloaded_files = []
        sidecar_suffix = "feather"
        if sidecar_name != "":
            sidecar_suffix = f"{sidecar_name}.feather"
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            futures = []
            for key in tqdm(self._manifest["key"].to_pylist()):
                sidecar_path = self.tile_destination / f"{key}.{sidecar_suffix}"
                sidecar_url = (
                    self.dataset.atlas_api_path
                    + f"/v1/project/{self.dataset.id}/index/projection/{self.id}/quadtree/{key}.{sidecar_suffix}"
                )
                futures.append(
                    ex.submit(
                        download_feather, sidecar_url, sidecar_path, headers=self.dataset.header, overwrite=overwrite
                    )
                )
                downloaded_files.append(sidecar_path)
            for f in futures:
                f.result()
        return downloaded_files

    @property
    def tile_destination(self):
        return Path("~/.nomic/cache", self.id).expanduser()

    @property
    def datum_id_field(self):
        return self.dataset.meta["unique_id_field"]

    def _get_atoms(self, ids: List[str]) -> List[Dict]:
        """
        Retrieves atoms by id

        Args:
            ids: list of atom ids

        Returns:
            A dictionary containing the resulting atoms, keyed by atom id.

        """

        if not isinstance(ids, list):
            raise ValueError("You must specify a list of ids when getting data.")

        response = requests.post(
            self.dataset.atlas_api_path + "/v1/project/atoms/get",
            headers=self.dataset.header,
            json={"project_id": self.dataset.id, "index_id": self.atlas_index_id, "atom_ids": ids},
        )

        if response.status_code == 200:
            return response.json()["atoms"]
        else:
            raise Exception(response.text)


class AtlasDataStream(AtlasClass):
    def __init__(self, name: Optional[str] = "contrastors"):
        super().__init__()
        if name != "contrastors":
            raise NotImplementedError("Only contrastors datastream is currently supported")
        self.name = name

    # TODO: add support for other datastreams
    def get_credentials(self):
        endpoint = f"/v1/data/{self.name}"
        response = requests.get(
            self.atlas_api_path + endpoint,
            headers=self.header,
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(response.text)


class AtlasDataset(AtlasClass):
    def __init__(
        self,
        identifier: Optional[str] = None,
        description: Optional[str] = "A description for your map.",
        unique_id_field: Optional[str] = None,
        is_public: bool = True,
        dataset_id=None,
        organization_name=None,
    ):
        """
        Creates or loads an AtlasDataset.
        AtlasDataset's store data (text, embeddings, etc) that you can organize by building indices.
        If the organization already contains a dataset with this name, it will be returned instead.

        **Parameters:**

        * **identifier** - The dataset identifier in the form `dataset` or `organization/dataset`. If no organization is passed, your default organization will be used.
        * **description** - A description for the dataset.
        * **unique_id_field** - The field that uniquely identifies each data point.
        * **is_public** - Should this dataset be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.
        * **dataset_id** - An alternative way to load a dataset is by passing the dataset_id directly. This only works if a dataset exists.
        """
        assert identifier is not None or dataset_id is not None, "You must pass a dataset identifier"
        # Normalize identifier.
        if identifier is not None:
            s = identifier.split("/", 1)
            identifier = unicodedata.normalize("NFD", s[-1])  # normalize accents
            identifier = identifier.lower().replace(" ", "-").replace("_", "-")
            identifier = re.sub(r"[^a-z0-9-]", "", identifier)
            identifier = re.sub(r"-+", "-", identifier)
            if len(s) == 2:
                identifier = f"{s[0]}/{identifier}"

        super().__init__()

        if organization_name is not None:
            raise DeprecationWarning(
                f"Passing organization_name has been removed in Nomic Python client 3.0. Instead identify your dataset with `organization_name/project_name` (e.g. sterling-cooper/november-ads)."
            )

        # Set this before possible early return.
        self._schema = None
        if dataset_id is not None:
            self.meta = self._get_project_by_id(dataset_id)
            return

        if not self.is_valid_dataset_identifier(identifier=str(identifier)):
            default_org_slug = self._get_current_users_main_organization()["slug"]
            identifier = default_org_slug + "/" + identifier

        dataset = self._get_dataset_by_slug_identifier(identifier=str(identifier))

        if dataset:  # dataset already exists
            logger.info(f"Loading existing dataset `{identifier}`.")
            dataset_id = dataset["id"]

        if dataset_id is None:  # if there is no existing project, make a new one.
            if unique_id_field is None:  # if not all parameters are specified, we weren't trying to make a project
                raise ValueError(f"Dataset `{identifier}` does not exist.")

            # if modality is None:
            #     raise ValueError("You must specify a modality when creating a new dataset.")
            #
            # assert modality in ['text', 'embedding'], "Modality must be either `text` or `embedding`"
            assert identifier is not None

            dataset_id = self._create_project(
                identifier=identifier,
                description=description,
                unique_id_field=unique_id_field,
                is_public=is_public,
            )

        self.meta = self._get_project_by_id(project_id=dataset_id)

    def delete(self):
        """
        Deletes an atlas dataset with all associated metadata.
        """
        organization = self._get_current_users_main_organization()
        organization_slug = organization["slug"]

        logger.info(f"Deleting dataset `{self.slug}` from organization `{organization_slug}`")

        self._delete_project_by_id(project_id=self.id)

        return False

    def _create_project(
        self,
        identifier: str,
        description: Optional[str],
        unique_id_field: str,
        is_public: bool = True,
    ):
        """
        Creates an Atlas Dataset.
        Atlas Datasets store data (text, embeddings, etc) that you can organize by building indices.
        If the organization already contains a dataset with this name, it will be returned instead.

        **Parameters:**

        * **identifier** - The identifier for the dataset.
        * **description** - A description for the dataset.
        * **unique_id_field** - The field that uniquely identifies each datum. If a datum does not contain this field, it will be added and assigned a random unique ID.
        * **is_public** - Should this dataset be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.

        **Returns:** project_id on success.

        """

        organization_id = self._get_organization_by_slug(slug=identifier)
        project_slug = identifier.split("/")[1]

        if "/" in identifier:
            org_name = identifier.split("/")[0]
            logger.info(f"Organization name: `{org_name}`")
        # supported_modalities = ['text', 'embedding']
        # if modality not in supported_modalities:
        #     msg = 'Tried to create dataset with modality: {}, but Atlas only supports: {}'.format(
        #         modality, supported_modalities
        #     )
        #     raise ValueError(msg)

        if unique_id_field is None:
            raise ValueError("You must specify a unique id field")
        if description is None:
            description = ""
        response = requests.post(
            self.atlas_api_path + "/v1/project/create",
            headers=self.header,
            json={
                "organization_id": organization_id,
                "project_name": project_slug,
                "description": description,
                "unique_id_field": unique_id_field,
                # 'modality': modality,
                "is_public": is_public,
            },
        )

        if response.status_code != 201:
            raise Exception(f"Failed to create dataset: {response.json()}")

        logger.info(f"Creating dataset `{response.json()['slug']}`")

        return response.json()["project_id"]

    def _latest_dataset_state(self):
        """
        Refreshes the project's state. Try to call this sparingly but use it when you need it.
        """

        self.meta = self._get_project_by_id(self.id)
        return self

    @property
    def indices(self) -> List[AtlasIndex]:
        self._latest_dataset_state()
        output = []
        for index in self.meta["atlas_indices"]:
            projections = []
            for projection in index["projections"]:
                projection = AtlasProjection(
                    dataset=self, projection_id=projection["id"], atlas_index_id=index["id"], name=index["index_name"]
                )
                projections.append(projection)
            index = AtlasIndex(
                atlas_index_id=index["id"],
                name=index["index_name"],
                indexed_field=index["indexed_field"],
                projections=projections,
            )
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
        """The UUID of the dataset."""
        return self.meta["id"]

    @property
    def id_field(self) -> str:
        return self.meta["unique_id_field"]

    @property
    def created_timestamp(self) -> datetime:
        return datetime.strptime(self.meta["created_timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z")

    @property
    def total_datums(self) -> int:
        """The total number of data points in the dataset."""
        return self.meta["total_datums_in_project"]

    @property
    def modality(self) -> str:
        return self.meta["modality"]

    @property
    def name(self) -> str:
        """The customizable name of the dataset."""
        return self.meta["project_name"]

    @property
    def slug(self) -> str:
        """The URL-safe identifier for this dataset."""
        return self.meta["slug"]

    @property
    def identifier(self) -> str:
        """The Atlas globally unique, URL-safe identifier for this dataset"""
        return self.meta["organization_slug"] + "/" + self.meta["slug"]

    @property
    def description(self):
        return self.meta["description"]

    @property
    def dataset_fields(self):
        return self.meta["project_fields"]

    @property
    def is_locked(self) -> bool:
        self._latest_dataset_state()
        return self.meta["insert_update_delete_lock"]

    @property
    def schema(self) -> Optional[pa.Schema]:
        if self._schema is not None:
            return self._schema
        if "schema" in self.meta and self.meta["schema"] is not None:
            self._schema: pa.Schema = ipc.read_schema(io.BytesIO(base64.b64decode(self.meta["schema"])))
            return self._schema
        return None

    @property
    def is_accepting_data(self) -> bool:
        """
        Checks if the dataset can accept data. Datasets cannot accept data when they are being indexed.

        Returns:
            True if dataset is unlocked for data additions, false otherwise.
        """
        return not self.is_locked

    @contextmanager
    def wait_for_dataset_lock(self):
        """Blocks thread execution until dataset is in a state where it can ingest data."""
        has_logged = False
        while True:
            if self.is_accepting_data:
                yield self
                break
            if not has_logged:
                logger.info(f"{self.identifier}: Waiting for dataset lock Release.")
                has_logged = True
            time.sleep(5)

    def get_map(
        self, name: Optional[str] = None, atlas_index_id: Optional[str] = None, projection_id: Optional[str] = None
    ) -> AtlasProjection:
        """
        Retrieves a map.

        Args:
            name: The name of your map. This defaults to your dataset name but can be different if you build multiple maps in your dataset.
            atlas_index_id: If specified, will only return a map if there is one built under the index with the id atlas_index_id.
            projection_id: If projection_id is specified, will only return a map if there is one built under the index with id projection_id.

        Returns:
            The map or a ValueError.
        """

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

        raise ValueError(f"Could not find a map named {name} in your dataset.")

    def create_index(
        self,
        name: Optional[str] = None,
        indexed_field: Optional[str] = None,
        modality: Optional[str] = None,
        projection: Union[bool, Dict, NomicProjectOptions] = True,
        topic_model: Union[bool, Dict, NomicTopicOptions] = True,
        duplicate_detection: Union[bool, Dict, NomicDuplicatesOptions] = True,
        embedding_model: Optional[Union[str, Dict, NomicEmbedOptions]] = None,
        reuse_embeddings_from_index: Optional[str] = None,
    ) -> Optional[AtlasProjection]:
        """
        Creates an index in the specified dataset.

        Args:
            name: The name of the index and the map.
            indexed_field: For text datasets, name the data field corresponding to the text to be mapped.
            reuse_embeddings_from_index: the name of the index to reuse embeddings from.
            modality: The data modality of this index. Currently, Atlas supports either `text`, `image`, or `embedding` indices.
            projection: Options for configuring the 2D projection algorithm
            topic_model: Options for configuring the topic model
            duplicate_detection: Options for configuring semantic duplicate detection
            embedding_model: Options for configuring the embedding model

        Returns:
            The projection this index has built.

        """

        self._latest_dataset_state()

        if isinstance(projection, Dict):
            projection = NomicProjectOptions(**projection)
        else:
            projection = NomicProjectOptions()

        topic_model_was_false = topic_model is False
        if isinstance(topic_model, Dict):
            topic_model = NomicTopicOptions(**topic_model)
        elif isinstance(topic_model, NomicTopicOptions):
            pass
        elif topic_model:
            topic_model = NomicTopicOptions(topic_label_field=indexed_field)
        else:
            topic_model = NomicTopicOptions(build_topic_model=False)

        if isinstance(duplicate_detection, Dict):
            duplicate_detection = NomicDuplicatesOptions(**duplicate_detection)
        elif isinstance(duplicate_detection, NomicDuplicatesOptions):
            pass
        elif duplicate_detection:
            duplicate_detection = NomicDuplicatesOptions()
        else:
            duplicate_detection = NomicDuplicatesOptions(tag_duplicates=False)

        if isinstance(embedding_model, Dict):
            embedding_model = NomicEmbedOptions(**embedding_model)
        elif isinstance(embedding_model, NomicEmbedOptions):
            pass
        elif isinstance(embedding_model, str):
            embedding_model = NomicEmbedOptions(model=embedding_model)  # type: ignore
        else:
            embedding_model = NomicEmbedOptions()

        if modality is None:
            modality = self.meta["modality"]

        if modality == "image":
            indexed_field = "_blob_hash"
            if indexed_field is not None:
                logger.warning("Ignoring indexed_field for image datasets. Only _blob_hash is supported.")

        colorable_fields = []

        for field in self.dataset_fields:
            if field not in [self.id_field, indexed_field] and not field.startswith("_"):
                colorable_fields.append(field)

        build_template = {}
        if modality == "embedding":
            if (not topic_model_was_false) and topic_model.topic_label_field is None:
                logger.warning(
                    "You did not specify the `topic_label_field` option in your topic_model, your dataset will not contain auto-labeled topics."
                )
            build_template = {
                "project_id": self.id,
                "index_name": name,
                "indexed_field": None,
                "atomizer_strategies": None,
                "model": None,
                "colorable_fields": colorable_fields,
                "model_hyperparameters": None,
                "nearest_neighbor_index": "HNSWIndex",
                "nearest_neighbor_index_hyperparameters": json.dumps({"space": "l2", "ef_construction": 100, "M": 16}),
                "projection": "NomicProject",
                "projection_hyperparameters": json.dumps(
                    {
                        "n_neighbors": projection.n_neighbors,
                        "n_epochs": projection.n_epochs,
                        "spread": projection.spread,
                        "local_neighborhood_size": projection.local_neighborhood_size,
                        "rho": projection.rho,
                        "model": projection.model,
                    }
                ),
                "topic_model_hyperparameters": json.dumps(
                    {
                        "build_topic_model": topic_model.build_topic_model,
                        "community_description_target_field": topic_model.topic_label_field,  # TODO change key to topic_label_field post v0.0.85
                        "cluster_method": topic_model.cluster_method,
                        "enforce_topic_hierarchy": topic_model.enforce_topic_hierarchy,
                    }
                ),
                "duplicate_detection_hyperparameters": json.dumps(
                    {
                        "tag_duplicates": duplicate_detection.tag_duplicates,
                        "duplicate_cutoff": duplicate_detection.duplicate_cutoff,
                    }
                ),
            }

        elif modality == "text" or modality == "image":
            # find the index id of the index with name reuse_embeddings_from_index
            reuse_embedding_from_index_id = None
            indices = self.indices
            if reuse_embeddings_from_index is not None:
                for index in indices:
                    if index.name == reuse_embeddings_from_index:
                        reuse_embedding_from_index_id = index.id
                        break
                if reuse_embedding_from_index_id is None:
                    raise Exception(
                        f"Could not find the index '{reuse_embeddings_from_index}' to re-use from. Possible options are {[index.name for index in indices]}"
                    )

            if indexed_field is None and modality == "text":
                raise Exception("You did not specify a field to index. Specify an 'indexed_field'.")

            if indexed_field not in self.dataset_fields:
                raise Exception(f"Indexing on {indexed_field} not allowed. Valid options are: {self.dataset_fields}")

            if modality == "image":
                if topic_model.topic_label_field is None:
                    print(
                        "You did not specify the `topic_label_field` option in your topic_model, your dataset will not contain auto-labeled topics."
                    )
                    topic_field = None
                    topic_model.build_topic_model = False
                else:
                    topic_field = (
                        topic_model.topic_label_field if topic_model.topic_label_field != indexed_field else None
                    )
            else:
                topic_field = topic_model.topic_label_field

            build_template = {
                "project_id": self.id,
                "index_name": name,
                "indexed_field": indexed_field,
                "atomizer_strategies": ["document", "charchunk"],
                "model": embedding_model.model,
                "colorable_fields": colorable_fields,
                "reuse_atoms_and_embeddings_from": reuse_embedding_from_index_id,
                "model_hyperparameters": json.dumps(
                    {
                        "dataset_buffer_size": 1000,
                        "batch_size": 20,
                        "polymerize_by": "charchunk",
                        "norm": "both",
                    }
                ),
                "nearest_neighbor_index": "HNSWIndex",
                "nearest_neighbor_index_hyperparameters": json.dumps({"space": "l2", "ef_construction": 100, "M": 16}),
                "projection": "NomicProject",
                "projection_hyperparameters": json.dumps(
                    {
                        "n_neighbors": projection.n_neighbors,
                        "n_epochs": projection.n_epochs,
                        "spread": projection.spread,
                        "local_neighborhood_size": projection.local_neighborhood_size,
                        "rho": projection.rho,
                        "model": projection.model,
                    }
                ),
                "topic_model_hyperparameters": json.dumps(
                    {
                        "build_topic_model": topic_model.build_topic_model,
                        "community_description_target_field": topic_field,
                        "cluster_method": topic_model.build_topic_model,
                        "enforce_topic_hierarchy": topic_model.enforce_topic_hierarchy,
                    }
                ),
                "duplicate_detection_hyperparameters": json.dumps(
                    {
                        "tag_duplicates": duplicate_detection.tag_duplicates,
                        "duplicate_cutoff": duplicate_detection.duplicate_cutoff,
                    }
                ),
            }

        response = requests.post(
            self.atlas_api_path + "/v1/project/index/create",
            headers=self.header,
            json=build_template,
        )
        if response.status_code != 200:
            logger.info("Create dataset failed with code: {}".format(response.status_code))
            logger.info("Additional info: {}".format(response.text))
            raise Exception(response.json()["detail"])

        job_id = response.json()["job_id"]

        job = requests.get(
            self.atlas_api_path + f"/v1/project/index/job/{job_id}",
            headers=self.header,
        ).json()

        index_id = job["index_id"]

        try:
            atlas_projection = self.get_map(atlas_index_id=index_id)
        except ValueError:
            # give some delay
            time.sleep(5)
            try:
                atlas_projection = self.get_map(atlas_index_id=index_id)
            except ValueError:
                atlas_projection = None

        if atlas_projection is None:
            logger.warning("Could not find a map being built for this dataset.")
        else:
            logger.info(
                f"Created map `{atlas_projection.name}` in dataset `{self.identifier}`: {atlas_projection.dataset_link}"
            )
        return atlas_projection

    def __repr__(self):
        m = self.meta
        return f"AtlasDataset: <{m}>"

    def _repr_html_(self):
        self._latest_dataset_state()
        m = self.meta
        html = f"""
            <strong><a href="https://atlas.nomic.ai/data/project/{m['id']}">{m['slug']}</strong></a>
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
                state = projection._status["index_build_stage"]
                if state == "Completed":
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
        """
        Retrieve the contents of the data given ids.

        Args:
            ids: a list of datum ids

        Returns:
            A list of dictionaries corresponding to the data.

        """

        if not isinstance(ids, list):
            raise ValueError("You must specify a list of ids when getting data.")
        if isinstance(ids[0], list):
            raise ValueError("You must specify a list of ids when getting data, not a nested list.")
        response = requests.post(
            self.atlas_api_path + "/v1/project/data/get",
            headers=self.header,
            json={"project_id": self.id, "datum_ids": ids},
        )

        if response.status_code == 200:
            return [item for item in response.json()["datums"]]
        else:
            raise Exception(response.text)

    def delete_data(self, ids: List[str]) -> bool:
        """
        Deletes the specified datapoints from the dataset.

        Args:
            ids: A list of data ids to delete

        Returns:
            True if data deleted successfully.

        """
        if not isinstance(ids, list):
            raise ValueError("You must specify a list of ids when deleting datums.")

        response = requests.post(
            self.atlas_api_path + "/v1/project/data/delete",
            headers=self.header,
            json={"project_id": self.id, "datum_ids": ids},
        )

        if response.status_code == 200:
            return True
        else:
            raise Exception(response.text)

    def add_data(
        self,
        data=Union[DataFrame, List[Dict], pa.Table],
        embeddings: Optional[np.ndarray] = None,
        blobs: Optional[List[Union[str, bytes, Image.Image]]] = None,
        pbar=None,
    ):
        """
        Adds data of varying modality to an Atlas dataset.
        Args:
            data: A pandas DataFrame, list of dictionaries, or pyarrow Table matching the dataset schema.
            embeddings: A numpy array of embeddings: each row corresponds to a row in the table. Use if you already have embeddings for your datapoints.
            blobs: A list of image paths, bytes, or PIL Images. Use if you want to create an AtlasDataset using image embeddings over your images. Note: Blobs are stored locally only.
            pbar: (Optional). A tqdm progress bar to update.
        """
        if isinstance(data, DataFrame):
            cols_before = set(data.columns)
            for col in cols_before:
                if col.startswith("_"):
                    raise ValueError(
                        f"You are attempting to upload a pandas dataframe with the column name {col}, but columns beginning with '_' are reserved for Atlas internal use. Please rename your column and try again."
                    )
            data = pa.Table.from_pandas(data)
            for newcol in set(data.column_names).difference(cols_before):
                logger.warning(f"Dropping column {newcol} added in pandas conversion to pyarrow")
                data = data.drop([newcol])

        if embeddings is not None:
            self._add_embeddings(data=data, embeddings=embeddings, pbar=pbar)
        elif isinstance(data, pa.Table) and "_embeddings" in data.column_names:  # type: ignore
            embeddings = np.array(data.column("_embeddings").to_pylist())  # type: ignore
            self._add_embeddings(data=data, embeddings=embeddings, pbar=pbar)
        elif blobs is not None:
            self._add_blobs(data=data, blobs=blobs, pbar=pbar)
        else:
            self._add_text(data=data, pbar=pbar)

    def _add_blobs(
        self, data: Union[DataFrame, List[Dict], pa.Table], blobs: List[Union[str, bytes, Image.Image]], pbar=None
    ):
        """
        Add data, with associated blobs, to the dataset.
        Uploads blobs to the server and associates them with the data.
        Blobs must reference objects stored locally
        """
        if isinstance(data, DataFrame):
            data = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            data = pa.Table.from_pylist(data)
        elif not isinstance(data, pa.Table):
            raise ValueError("Data must be a pandas DataFrame, list of dictionaries, or a pyarrow Table.")

        blob_upload_endpoint = "/v1/project/data/add/blobs"

        # uploda batch of blobs
        # return hash of blob
        # add hash to data as _blob_hash
        # set indexed_field to _blob_hash
        # call _add_data

        # Cast self id field to string for merged data lower down on function
        data = data.set_column(  # type: ignore
            data.schema.get_field_index(self.id_field), self.id_field, pc.cast(data[self.id_field], pa.string())  # type: ignore
        )

        ids = data[self.id_field].to_pylist()  # type: ignore
        if not isinstance(ids[0], str):
            ids = [str(uuid) for uuid in ids]

        # TODO: add support for other modalities
        images = []
        for uuid, blob in tqdm(zip(ids, blobs), total=len(ids), desc="Loading images"):
            if (isinstance(blob, str) or isinstance(blob, Path)) and os.path.exists(blob):
                # Auto resize to max 512x512
                image = Image.open(blob)
                image = image.convert("RGB")
                if image.height > 512 or image.width > 512:
                    image = image.resize((512, 512))
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                images.append((uuid, buffered.getvalue()))
            elif isinstance(blob, bytes):
                images.append((uuid, blob))
            elif isinstance(blob, Image.Image):
                blob = blob.convert("RGB")  # type: ignore
                if blob.height > 512 or blob.width > 512:
                    blob = blob.resize((512, 512))
                buffered = BytesIO()
                blob.save(buffered, format="JPEG")
                images.append((uuid, buffered.getvalue()))
            else:
                raise ValueError(f"Invalid blob type for {uuid}. Must be a path to an image, bytes, or PIL Image.")

        batch_size = 40
        num_workers = 10

        def send_request(i):
            image_batch = images[i : i + batch_size]
            ids = [uuid for uuid, _ in image_batch]
            blobs = [("blobs", blob) for _, blob in image_batch]
            response = requests.post(
                self.atlas_api_path + blob_upload_endpoint,
                headers=self.header,
                data={"dataset_id": self.id},
                files=blobs,
            )
            if response.status_code != 200:
                raise Exception(response.text)
            return {uuid: blob_hash for uuid, blob_hash in zip(ids, response.json()["hashes"])}

        # if this method is being called internally, we pass a global progress bar
        if pbar is None:
            pbar = tqdm(total=len(data), desc="Uploading blobs to Atlas")

        hash_schema = pa.schema([(self.id_field, pa.string()), ("_blob_hash", pa.string())])
        returned_ids = []
        returned_hashes = []

        succeeded = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(send_request, i): i for i in range(0, len(data), batch_size)}

            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                # add hash to data as _blob_hash
                for uuid, blob_hash in response.items():
                    returned_ids.append(uuid)
                    returned_hashes.append(blob_hash)

                # A successful upload.
                succeeded += len(response)
                pbar.update(len(response))

        hash_tb = pa.Table.from_pydict({self.id_field: returned_ids, "_blob_hash": returned_hashes}, schema=hash_schema)
        merged_data = data.join(right_table=hash_tb, keys=self.id_field)  # type: ignore

        self._add_data(merged_data, pbar=pbar)

    def _add_text(self, data=Union[DataFrame, List[Dict], pa.Table], pbar=None):
        """
        Add text data to the dataset.
        data: A pandas DataFrame, a list of python dictionaries, or a pyarrow Table matching the dataset schema.
        pbar: (Optional). A tqdm progress bar to display progress.
        """
        if isinstance(data, DataFrame):
            data = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            data = pa.Table.from_pylist(data)
        elif not isinstance(data, pa.Table):
            raise ValueError("Data must be a pandas DataFrame, list of dictionaries, or a pyarrow Table.")
        self._add_data(data, pbar=pbar)

    def _add_embeddings(self, data: Union[DataFrame, List[Dict], pa.Table], embeddings: np.ndarray, pbar=None):
        """
        Add data, with associated embeddings, to the dataset.

        Args:
            data: A pandas DataFrame, list of dictionaries, or pyarrow Table matching the dataset schema.
            embeddings: A numpy array of embeddings: each row corresponds to a row in the table.
            pbar: (Optional). A tqdm progress bar to update.
        """

        """
        # TODO: validate embedding size.
        assert embeddings.shape[1] == self.embedding_size, "Embedding size must match the embedding size of the dataset."
        """
        assert type(embeddings) == np.ndarray, "Embeddings must be a NumPy array."
        assert len(embeddings.shape) == 2, "Embeddings must be a 2D NumPy array."
        assert len(data) == embeddings.shape[0], "Data and embeddings must have the same number of rows."
        assert len(data) > 0, "Data must have at least one row."

        tb: pa.Table

        if isinstance(data, DataFrame):
            tb = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            tb = pa.Table.from_pylist(data)
        elif isinstance(data, pa.Table):
            tb = data
        else:
            raise ValueError(
                f"Data must be a pandas DataFrame, list of dictionaries, or a pyarrow Table, not {type(data)}"
            )

        del data

        # Add embeddings to the data.
        # Allow 2d embeddings to stay at single-fp precision.
        if not (embeddings.shape[1] == 2 and embeddings.dtype == np.float32):
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
        """
        Low level interface to upload an Arrow Table. Users should generally call 'add_text' or 'add_embeddings.'

        Args:
            data: A pyarrow Table that will be cast to the dataset schema.
            pbar: A tqdm progress bar to update.
        Returns:
            None
        """

        # Exactly 10 upload workers at a time.

        num_workers = 10

        # Each worker currently is too slow beyond a shard_size of 10000

        # The heuristic here is: Never let shards be more than 10,000 items,
        # OR more than 16MB uncompressed. Whichever is smaller.

        bytesize = data.nbytes
        nrow = len(data)

        shard_size = 5_000
        n_chunks = int(np.ceil(nrow / shard_size))
        # Chunk into 16MB pieces. These will probably compress down a bit.
        if bytesize / n_chunks > 16_000_000:
            shard_size = int(np.ceil(nrow / (bytesize / 16_000_000)))

        data = self._validate_and_correct_arrow_upload(
            data=data,
            project=self,
        )

        upload_endpoint = "/v1/project/data/add/arrow"

        # Actually do the upload
        def send_request(i):
            data_shard = data.slice(i, shard_size)
            with io.BytesIO() as buffer:
                data_shard = data_shard.replace_schema_metadata({"project_id": self.id})
                feather.write_feather(data_shard, buffer, compression="zstd", compression_level=6)
                buffer.seek(0)

                response = requests.post(
                    self.atlas_api_path + upload_endpoint,
                    headers=self.header,
                    data=buffer,
                )
                return response

        # if this method is being called internally, we pass a global progress bar
        close_pbar = False
        if pbar is None:
            close_pbar = True
            pbar = tqdm(total=int(len(data)) // shard_size)
        failed = 0
        failed_reqs = 0
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
                            if "more datums exceeds your organization limit" in response.json():
                                return False
                            if "Project transaction lock is held" in response.json():
                                raise Exception(
                                    "Project is currently indexing and cannot ingest new datums. Try again later."
                                )
                            if "Insert failed due to ID conflict" in response.json():
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
                                    f"{self.identifier}: Connection failed for records {start_point}-{start_point + shard_size}, retrying."
                                )
                                failure_fraction = errors_504 / (failed + succeeded + errors_504)
                                if failure_fraction > 0.5 and errors_504 > shard_size * 3:
                                    raise RuntimeError(
                                        f"{self.identifier}: Atlas is under high load and cannot ingest datums at this time. Please try again later."
                                    )
                                new_submission = executor.submit(send_request, start_point)
                                futures[new_submission] = start_point
                                response.close()
                            else:
                                logger.error(f"{self.identifier}: Shard upload failed: {response}")
                                failed += shard_size
                                pbar.update(1)
                                response.close()
                            failed_reqs += 1
                            if failed_reqs > 10:
                                raise RuntimeError(
                                    f"{self.identifier}: Too many upload requests have failed at this time. Please try again later."
                                )
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

    def update_maps(self, data: List[Dict], embeddings: Optional[np.ndarray] = None, num_workers: int = 10):
        """
        Utility method to update a project's maps by adding the given data.

        Args:
            data: An [N,] element list of dictionaries containing metadata for each embedding.
            embeddings: An [N, d] matrix of embeddings for updating embedding dataset. Leave as None to update text dataset.
            shard_size: Data is uploaded in parallel by many threads. Adjust the number of datums to upload by each worker.
            num_workers: The number of workers to use when sending data.

        """

        raise DeprecationWarning(
            f"The function AtlasDataset.update_maps is deprecated. Use AtlasDataset.add_data() instead."
        )

    def update_indices(self, rebuild_topic_models: bool = False):
        """
        Rebuilds all maps in a dataset with the latest state dataset data state. Maps will not be rebuilt to
        reflect the additions, deletions or updates you have made to your data until this method is called.

        Args:
            rebuild_topic_models: (Default False) - If true, will create new topic models when updating these indices.
        """

        raise DeprecationWarning(
            f"The function AtlasDataset.update_indices is deprecated. Use AtlasDataset.add_data() instead."
        )
