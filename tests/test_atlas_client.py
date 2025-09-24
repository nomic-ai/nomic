import random
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from nomic import AtlasDataset, atlas
from nomic.data_inference import ProjectionOptions


def gen_random_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


def gen_temp_identifier() -> str:
    now = datetime.now().isoformat(timespec="seconds")
    rand = random.randint(0, 10000)
    return f"{now}-{rand}"


def test_integration_map_idless_embeddings():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 512)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    dataset = atlas.map_data(
        identifier=f"unittest-dataset-{gen_temp_identifier()}", embeddings=embeddings, data=data_payload
    )
    AtlasDataset(dataset.identifier).delete()


def test_integration_map_embeddings_with_errors():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)

    name = f"unittest-dataset-{gen_temp_identifier()}"
    # test nested dictionaries
    with pytest.raises(Exception):
        data = [{"key": {"nested_key": "nested_value"}} for i in range(len(embeddings))]
        dataset = atlas.map_data(embeddings=embeddings, data=data, identifier=name, is_public=True)

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass

    name = f"unittest-dataset-{gen_temp_identifier()}"
    # test underscore
    with pytest.raises(Exception):
        data = [{"__hello": {"hello"}} for i in range(len(embeddings))]
        dataset = atlas.map_data(embeddings=embeddings, data=data, identifier=name, is_public=True)

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass

    name = f"unittest-dataset-{gen_temp_identifier()}"
    # test to long ids
    with pytest.raises(Exception):
        dataset = atlas.map_data(
            embeddings=embeddings,
            data=data,
            identifier=name,
            is_public=True,
        )

        assert isinstance(dataset.created_timestamp, datetime)

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass


def test_integration_map_text_errors():
    # no indexed field
    name = f"unittest-dataset-{gen_temp_identifier()}"
    with pytest.raises(Exception):
        dataset = atlas.map_data(
            data=[{"key": "a"}],
            indexed_field="text",
            is_public=True,
            identifier=name,
            description="test map description",
        )

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass


def test_date_metadata():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [
        {"my_date": datetime(2022, 1, i), "my_random_date": gen_random_datetime()}
        for i in range(1, len(embeddings) + 1)
    ]

    dataset = atlas.map_data(
        embeddings=embeddings, identifier=f"unittest-dataset-{gen_temp_identifier()}", data=data, is_public=True
    )

    assert dataset.id

    dataset.delete()

    # put an invalid iso timestamp after the first valid isotimestamp , make sure the client fails
    with pytest.raises(Exception):
        data[1]["my_date"] = data[1]["my_date"] + "asdf"
        dataset = atlas.map_data(
            embeddings=embeddings,
            identifier=f"unittest-dataset-{gen_temp_identifier()}",
            data=data,
            is_public=True,
        )
        dataset.delete()


def test_topics():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ["foo", "bar", "baz", "bat"]
    dates = [datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1)]
    data = [{"upload": 0.0, "text": texts[i % 4], "date": dates[i % 3]} for i in range(len(embeddings))]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{gen_temp_identifier()}",
        data=data,
        is_public=True,
        topic_model=dict(topic_label_field="text"),
    )

    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)

    assert len(projection.topics.metadata) > 0
    assert isinstance(projection.topics.group_by_topic(topic_depth=1), list)

    dataset.delete()


def test_data():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ["foo", "bar", "baz", "bat"]
    data = [{"upload": 0.0, "text": str(i)} for i in range(len(embeddings))]
    all_columns = data[0].keys()

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{gen_temp_identifier()}",
        data=data,
        is_public=True,
        topic_model=dict(build_topic_model=True, community_description_target_field="text"),
    )

    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)

    df = projection.data.df
    assert len(df) > 0
    for column in all_columns:
        assert column in df.columns
    map_proj = dataset.maps[0]

    retrieved_embeddings = map_proj.embeddings.latent

    assert dataset.total_datums == num_embeddings
    assert retrieved_embeddings.shape[0] == num_embeddings

    assert len(map_proj.topics.df) == num_embeddings

    dataset.delete()


words = [
    "foo",
    "bar",
    "baz",
    "bat",
    "glorp",
    "gloop",
    "glib",
    "glub",
    "florp",
    "floop",
    "flib",
    "flub",
    "blorp",
    "bloop",
    "blib",
    "blub",
    "slorp",
    "sloop",
    "slib",
    "slub",
    "clorp",
    "cloop",
    "clib",
    "club",
    "plorp",
    "ploop",
    "plib",
    "plub",
    "zlorp",
    "zloop",
    "zlib",
    "zlub",
    "xlorp",
    "xloop",
    "xlib",
    "xlub",
    "vlorp",
    "vloop",
    "vlib",
    "vlub",
    "nlorp",
    "nloop",
    "nlib",
    "nlub",
    "mlorp",
    "mloop",
    "mlib",
    "mlub",
]


def test_flawed_ids():
    """
    Check that null and empty strings do not block an index build.
    """
    p = AtlasDataset(f"unittest-dataset-{gen_temp_identifier()}", unique_id_field="id")

    elements = []
    for i in range(10):
        if i % 3 == 0 and i % 5 == 0:
            elements.append({"text": "fizzbuzz", "id": str(i)})
        elif i % 3 == 0:
            elements.append({"text": "fizz", "id": str(i)})
        elif i % 5 == 0:
            elements.append({"text": "buzz", "id": str(i)})
    p.add_data(data=elements)
    with pytest.raises(ValueError):
        p.add_data(data=[{"text": "fizzbuzz", "id": None}])
    with pytest.raises(ValueError):
        p.add_data(data=[{"text": "fizzbuzz", "id": "A" * 100}])
    p.delete()


def test_weird_inputs():
    """
    Check that null and empty strings do not block an index build.
    """
    dataset = AtlasDataset(f"unittest-dataset-{gen_temp_identifier()}")

    elements = []
    for i in range(20):
        if i % 3 == 0 and i % 5 == 0:
            elements.append({"text": "fizzbuzz"})
        elif i % 3 == 0:
            elements.append({"text": "fizz"})
        elif i % 5 == 0:
            elements.append({"text": "buzz"})
        elif i % 7 == 0:
            elements.append({"text": None})
        elif i % 2 == 0:
            elements.append({"text": ""})
        else:
            elements.append({"text": "foo"})
    dataset.add_data(data=elements)
    projection = dataset.create_index(indexed_field="text", topic_model=True)
    if projection:
        _wait_for_projection_completion(projection)
        assert True
    else:
        pytest.fail("Could not find projection to wait for in test_weird_inputs")

    dataset.delete()


def test_integration_map_embeddings():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{gen_temp_identifier()}",
        data=data_payload,
        is_public=True,
    )

    map_proj = dataset.maps[0]

    _wait_for_projection_completion(map_proj)

    retrieved_embeddings = map_proj.embeddings.latent

    assert dataset.total_datums == num_embeddings
    assert retrieved_embeddings.shape[0] == num_embeddings

    assert len(map_proj.topics.df) == num_embeddings

    assert isinstance(map_proj.topics.hierarchy, dict)

    for p in dataset.projections:
        assert p.map_link

    dataset.delete()


def test_integration_map_text_pandas():
    size = 50
    data = pd.DataFrame(
        {
            "color": [random.choice(["red", "blue", "green"]) for i in range(size)],
        }
    )

    dataset = atlas.map_data(
        identifier=f"UNITTEST_pandas_text-{gen_temp_identifier()}", indexed_field="color", data=data, is_public=True
    )

    assert dataset.total_datums == 50

    dataset.delete()


def test_integration_map_text_arrow():
    size = 50
    data = pa.Table.from_pydict(
        {
            "color": [random.choice(["red", "blue", "green"]) for i in range(size)],
        }
    )

    dataset = atlas.map_data(
        identifier=f"UNITTEST_arrow_text-{gen_temp_identifier()}",
        indexed_field="color",
        data=data,
        is_public=True,
    )

    assert dataset.total_datums == 50

    dataset.delete()


def test_contrastors_access():
    keys = atlas._get_datastream_credentials()
    assert keys["access_key"] is not None
    assert keys["secret_key"] is not None


def _wait_for_projection_completion(projection, timeout_seconds=600):
    """Polls the projection status until it's 'Done' or timeout."""
    start_time = time.time()
    printed_waiting = False
    final_status = "Unknown"
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            try:
                last_status_data = projection._status
                final_status = (
                    last_status_data.get("index_build_stage")
                    or last_status_data.get("job_state")
                    or "ErrorFetchingStatus"
                )
            except Exception as e:
                final_status = f"Error fetching status: {e}"
            raise TimeoutError(
                f"Timed out waiting for projection {projection.id} to complete after {timeout_seconds} seconds. Last status: {final_status}"
            )

        try:
            status = projection._status
            build_stage = status.get("index_build_stage")
            if build_stage is None:
                build_stage = status.get("job_state")
            final_status = build_stage

            if build_stage == "Done" or build_stage == "Completed":
                total_time = time.time() - start_time
                print(f"Projection {projection.id} completed in {total_time:.2f} seconds.")
                return
            else:
                if not printed_waiting:
                    print(f"Waiting for projection {projection.id} to complete. Current status: {build_stage}")
                    printed_waiting = True
                time.sleep(10)
        except Exception as e:
            print(f"Error fetching status for projection {projection.id}: {e}. Retrying...")
            if not printed_waiting:
                printed_waiting = True
            time.sleep(10)


def test_integration_map_embeddings_explicit_umap():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    dataset_identifier = f"unittest-umap-explicit-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection=ProjectionOptions(model="umap", n_neighbors=5, min_dist=0.01, n_epochs=25),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


@patch("tests.test_atlas_client._wait_for_projection_completion")
@patch("nomic.atlas.map_data")
def test_map_embeddings_explicit_umap(mock_map_data, mock_wait_for_completion):
    # Setup mock_map_data
    mock_dataset = MagicMock(spec=AtlasDataset)
    mock_dataset.id = "mock_dataset_id"

    mock_projection_obj = (
        MagicMock()
    )  # Renamed to avoid conflict with 'projection' variable name from NomicProjectOptions
    mock_dataset.maps = [mock_projection_obj]
    mock_map_data.return_value = mock_dataset

    # Original test logic's data setup
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    dataset_identifier = f"unittest-umap-explicit-{gen_temp_identifier()}"
    umap_options = ProjectionOptions(model="umap", n_neighbors=5, min_dist=0.01, n_epochs=25)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    # Call the function that would normally call atlas.map_data
    dataset_out = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection=umap_options,
    )

    # Assertions on map_data call
    called_args, called_kwargs = mock_map_data.call_args

    assert called_kwargs["identifier"] == dataset_identifier
    assert np.array_equal(called_kwargs["embeddings"], embeddings)
    assert called_kwargs["data"] == data_payload
    assert called_kwargs["is_public"] is True
    actual_projection_options = called_kwargs["projection"]
    assert isinstance(actual_projection_options, ProjectionOptions)
    assert actual_projection_options.model == umap_options.model
    assert actual_projection_options.n_neighbors == umap_options.n_neighbors
    assert actual_projection_options.min_dist == umap_options.min_dist
    assert actual_projection_options.n_epochs == umap_options.n_epochs
    mock_map_data.assert_called_once()

    assert dataset_out.id == "mock_dataset_id"

    # Call dependent logic
    projection_out = dataset_out.maps[0]
    _wait_for_projection_completion(projection_out)

    # Assert that our mock _wait_for_projection_completion was called
    mock_wait_for_completion.assert_called_once_with(mock_projection_obj)

    dataset_out.delete()
    mock_dataset.delete.assert_called_once()


def test_integration_map_text_explicit_umap():
    size = 50
    data = pd.DataFrame(
        {
            "text_field": [words[i % len(words)] for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-text-umap-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        indexed_field="text_field",
        data=data,
        is_public=True,
        projection=ProjectionOptions(model="umap", n_neighbors=5, min_dist=0.05, n_epochs=30),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


@patch("tests.test_atlas_client._wait_for_projection_completion")
@patch("nomic.atlas.map_data")
def test_map_text_explicit_umap(mock_map_data, mock_wait_for_completion):
    mock_dataset = MagicMock(spec=AtlasDataset)
    mock_dataset.id = "mock_dataset_id_text"
    mock_projection_obj = MagicMock()
    mock_dataset.maps = [mock_projection_obj]
    mock_map_data.return_value = mock_dataset
    size = 50
    data_df = pd.DataFrame(
        {
            "text_field": [words[i % len(words)] for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-text-umap-{gen_temp_identifier()}"
    umap_options = ProjectionOptions(model="umap", n_neighbors=5, min_dist=0.05, n_epochs=30)

    # Call the function
    dataset_out = atlas.map_data(
        identifier=dataset_identifier,
        indexed_field="text_field",
        data=data_df,
        is_public=True,
        projection=umap_options,
    )

    # Assertions on map_data call
    called_args, called_kwargs = mock_map_data.call_args

    assert called_kwargs["identifier"] == dataset_identifier
    assert called_kwargs["indexed_field"] == "text_field"
    assert called_kwargs["data"].equals(data_df)  # Using .equals for DataFrame comparison
    assert called_kwargs["is_public"] is True
    actual_projection_options = called_kwargs["projection"]
    assert isinstance(actual_projection_options, ProjectionOptions)
    assert actual_projection_options.model == umap_options.model
    assert actual_projection_options.n_neighbors == umap_options.n_neighbors
    assert actual_projection_options.min_dist == umap_options.min_dist
    assert actual_projection_options.n_epochs == umap_options.n_epochs
    mock_map_data.assert_called_once()

    assert dataset_out.id == "mock_dataset_id_text"

    projection_out = dataset_out.maps[0]
    _wait_for_projection_completion(projection_out)

    mock_wait_for_completion.assert_called_once_with(mock_projection_obj)

    dataset_out.delete()
    mock_dataset.delete.assert_called_once()


def test_integration_map_embeddings_explicit_nomic_project():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    dataset_identifier = f"unittest-nomic-explicit-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection=ProjectionOptions(model="nomic-project-v1", n_neighbors=7, n_epochs=20, spread=0.6),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


@patch("tests.test_atlas_client._wait_for_projection_completion")
@patch("nomic.atlas.map_data")
def test_map_embeddings_explicit_nomic_project(mock_map_data, mock_wait_for_completion):
    mock_dataset = MagicMock(spec=AtlasDataset)
    mock_dataset.id = "mock_dataset_id_nomic_embed"
    mock_projection_obj = MagicMock()
    mock_dataset.maps = [mock_projection_obj]
    mock_map_data.return_value = mock_dataset

    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    dataset_identifier = f"unittest-nomic-explicit-{gen_temp_identifier()}"
    nomic_options = ProjectionOptions(model="nomic-project-v1", n_neighbors=7, n_epochs=20, spread=0.6)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    dataset_out = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection=nomic_options,
    )

    called_args, called_kwargs = mock_map_data.call_args
    assert called_kwargs["identifier"] == dataset_identifier
    assert np.array_equal(called_kwargs["embeddings"], embeddings)
    assert called_kwargs["data"] == data_payload
    assert called_kwargs["is_public"] is True

    actual_projection_options = called_kwargs["projection"]
    assert isinstance(actual_projection_options, ProjectionOptions)
    assert actual_projection_options.n_neighbors == nomic_options.n_neighbors
    assert actual_projection_options.model == nomic_options.model
    assert actual_projection_options.n_epochs == nomic_options.n_epochs
    assert actual_projection_options.spread == nomic_options.spread
    mock_map_data.assert_called_once()

    assert dataset_out.id == mock_dataset.id
    projection_out = dataset_out.maps[0]
    _wait_for_projection_completion(projection_out)
    mock_wait_for_completion.assert_called_once_with(mock_projection_obj)

    dataset_out.delete()
    mock_dataset.delete.assert_called_once()


def test_integration_map_text_explicit_nomic_project():
    size = 50
    data = pd.DataFrame(
        {
            "text_field": [words[i % len(words)] for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-text-nomic-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        indexed_field="text_field",
        data=data,
        is_public=True,
        projection=ProjectionOptions(
            model="nomic-project-v2", n_epochs=30, spread=0.5, rho=0.5, local_neighborhood_size=32
        ),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


@patch("tests.test_atlas_client._wait_for_projection_completion")
@patch("nomic.atlas.map_data")
def test_map_text_explicit_nomic_project(mock_map_data, mock_wait_for_completion):
    mock_dataset = MagicMock(spec=AtlasDataset)
    mock_dataset.id = "mock_dataset_id_nomic_text"
    mock_projection_obj = MagicMock()
    mock_dataset.maps = [mock_projection_obj]
    mock_map_data.return_value = mock_dataset

    size = 50
    data_df = pd.DataFrame(
        {
            "text_field": [words[i % len(words)] for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-text-nomic-{gen_temp_identifier()}"
    nomic_options = ProjectionOptions(
        model="nomic-project-v2", n_epochs=30, spread=0.5, rho=0.5, local_neighborhood_size=32
    )

    dataset_out = atlas.map_data(
        identifier=dataset_identifier,
        indexed_field="text_field",
        data=data_df,
        is_public=True,
        projection=nomic_options,
    )

    called_args, called_kwargs = mock_map_data.call_args
    assert called_kwargs["identifier"] == dataset_identifier
    assert called_kwargs["indexed_field"] == "text_field"
    assert called_kwargs["data"].equals(data_df)
    assert called_kwargs["is_public"] is True

    actual_projection_options = called_kwargs["projection"]
    assert isinstance(actual_projection_options, ProjectionOptions)
    assert actual_projection_options.n_epochs == nomic_options.n_epochs
    assert actual_projection_options.spread == nomic_options.spread
    assert actual_projection_options.model == nomic_options.model
    assert actual_projection_options.rho == nomic_options.rho
    assert actual_projection_options.local_neighborhood_size == nomic_options.local_neighborhood_size
    mock_map_data.assert_called_once()

    assert dataset_out.id == mock_dataset.id
    projection_out = dataset_out.maps[0]
    _wait_for_projection_completion(projection_out)
    mock_wait_for_completion.assert_called_once_with(mock_projection_obj)

    dataset_out.delete()
    mock_dataset.delete.assert_called_once()


def test_integration_map_embeddings_auto_with_options():
    num_embeddings = 50  # Small dataset, backend should pick UMAP if it has logic for it
    embeddings = np.random.rand(num_embeddings, 10)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    dataset_identifier = f"unittest-auto-opts-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection={
            "model": "nomic-project-v1",
            "n_neighbors": 6,
            "min_dist": 0.02,
            "n_epochs": 22,
            "spread": 0.7,
        },
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


def test_integration_map_embeddings_legacy_dict_with_explicit_algorithm():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    data_payload = [{"description": "test_value"} for _ in range(num_embeddings)]

    # Test with explicit algorithm="umap"
    dataset_identifier = f"unittest-legacy-dict-algo-umap-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection={"model": "umap", "n_neighbors": 9, "min_dist": 0.3, "n_epochs": 27},
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()

    dataset_identifier = f"unittest-legacy-dict-algo-nomic-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        embeddings=embeddings,
        data=data_payload,
        is_public=True,
        projection={"model": "nomic-project-v2", "n_neighbors": 11, "n_epochs": 28},
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


@pytest.mark.skip(reason="Test is failing")
def test_integration_map_images():
    size = 30
    # Generate random PIL images
    images = []
    for _ in range(size):
        img = Image.new("RGB", (60, 30), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        images.append(img)

    data = pd.DataFrame(
        {
            "description": [f"This is image {i}" for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-images-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        data=data,
        blobs=images,
        is_public=True,
    )
    assert dataset.id is not None
    assert dataset.modality == "image"  # Or check based on how map_data sets it.

    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)

    # Additional assertions can be added here, e.g., checking dataset.total_datums
    assert dataset.total_datums == size

    dataset.delete()
