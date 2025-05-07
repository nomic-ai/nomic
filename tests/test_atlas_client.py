import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from nomic import AtlasDataset, atlas
from nomic.data_inference import NomicProjectOptions, UMAPOptions


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


def test_map_idless_embeddings():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 512)

    dataset = atlas.map_data(identifier=f"unittest-dataset-{gen_temp_identifier()}", embeddings=embeddings)
    AtlasDataset(dataset.identifier).delete()


def test_map_embeddings_with_errors():
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
        data = [{"id": str(uuid.uuid4()) + "a"} for i in range(len(embeddings))]
        dataset = atlas.map_data(
            embeddings=embeddings,
            data=data,
            identifier=name,
            id_field="id",
            is_public=True,
        )

        assert isinstance(dataset.created_timestamp, datetime)

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass


def test_map_text_errors():
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
            id_field="id",
            data=data,
            is_public=True,
        )
        dataset.delete()


def test_topics():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ["foo", "bar", "baz", "bat"]
    dates = [datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1)]
    data = [
        {"field": str(uuid.uuid4()), "id": str(uuid.uuid4()), "upload": 0.0, "text": texts[i % 4], "date": dates[i % 3]}
        for i in range(len(embeddings))
    ]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{gen_temp_identifier()}",
        id_field="id",
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
    data = [
        {"field": str(uuid.uuid4()), "id": str(uuid.uuid4()), "upload": 0.0, "text": str(i)}
        for i in range(len(embeddings))
    ]
    all_columns = data[0].keys()

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{gen_temp_identifier()}",
        id_field="id",
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
    dataset = AtlasDataset(f"unittest-dataset-{gen_temp_identifier()}", unique_id_field="id")

    elements = []
    for i in range(20):
        if i % 3 == 0 and i % 5 == 0:
            elements.append({"text": "fizzbuzz", "id": str(i)})
        elif i % 3 == 0:
            elements.append({"text": "fizz", "id": str(i)})
        elif i % 5 == 0:
            elements.append({"text": "buzz", "id": str(i)})
        elif i % 7 == 0:
            elements.append({"text": None, "id": str(i)})
        elif i % 2 == 0:
            elements.append({"text": "", "id": str(i)})
        else:
            elements.append({"text": "foo", "id": str(i)})
    dataset.add_data(data=elements)
    projection = dataset.create_index(indexed_field="text", topic_model=True)
    if projection:
        _wait_for_projection_completion(projection)
        assert True
    else:
        pytest.fail("Could not find projection to wait for in test_weird_inputs")

    dataset.delete()


def test_map_embeddings():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{"field": str(uuid.uuid4()), "id": str(uuid.uuid4())} for i in range(len(embeddings))]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{gen_temp_identifier()}",
        id_field="id",
        data=data,
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


def test_map_text_pandas():
    size = 50
    data = pd.DataFrame(
        {
            "field": [str(uuid.uuid4()) for i in range(size)],
            "color": [random.choice(["red", "blue", "green"]) for i in range(size)],
        }
    )

    dataset = atlas.map_data(
        identifier=f"UNITTEST_pandas_text-{gen_temp_identifier()}", indexed_field="color", data=data, is_public=True
    )

    assert dataset.total_datums == 50

    dataset.delete()


def test_map_text_arrow():
    size = 50
    data = pa.Table.from_pydict(
        {
            "field": [str(uuid.uuid4()) for i in range(size)],
            "id": [str(uuid.uuid4()) for i in range(size)],
            "color": [random.choice(["red", "blue", "green"]) for i in range(size)],
        }
    )

    dataset = atlas.map_data(
        identifier=f"UNITTEST_arrow_text-{gen_temp_identifier()}",
        id_field="id",
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


def test_map_embeddings_explicit_umap():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{"id": str(i)} for i in range(num_embeddings)]

    dataset_identifier = f"unittest-umap-explicit-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        embeddings=embeddings,
        data=data,
        is_public=True,
        projection=UMAPOptions(n_neighbors=5, min_dist=0.01, n_epochs=25),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


def test_map_text_explicit_umap():
    size = 50
    data = pd.DataFrame(
        {
            "text_field": [words[i % len(words)] for i in range(size)],
            "id": [str(i) for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-text-umap-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        indexed_field="text_field",
        data=data,
        is_public=True,
        projection=UMAPOptions(n_neighbors=5, min_dist=0.05, n_epochs=30),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


def test_map_embeddings_explicit_nomic_project():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{"id": str(i)} for i in range(num_embeddings)]

    dataset_identifier = f"unittest-nomic-explicit-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        embeddings=embeddings,
        data=data,
        is_public=True,
        projection=NomicProjectOptions(n_neighbors=7, model="nomic-project-v1", n_epochs=20, spread=0.6),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


def test_map_text_explicit_nomic_project():
    size = 50
    data = pd.DataFrame(
        {
            "text_field": [words[i % len(words)] for i in range(size)],
            "id": [str(i) for i in range(size)],
        }
    )
    dataset_identifier = f"unittest-text-nomic-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        indexed_field="text_field",
        data=data,
        is_public=True,
        projection=NomicProjectOptions(
            n_epochs=30, spread=0.5, model="nomic-project-v2", rho=0.5, local_neighborhood_size=32
        ),
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


def test_map_embeddings_auto_with_options():
    num_embeddings = 50  # Small dataset, backend should pick UMAP if it has logic for it
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{"id": str(i)} for i in range(num_embeddings)]

    dataset_identifier = f"unittest-auto-opts-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        embeddings=embeddings,
        data=data,
        is_public=True,
        projection={
            "algorithm": "auto",
            "n_neighbors": 6,
            "min_dist": 0.02,
            "n_epochs": 22,
            "model": "nomic-project-v1",
            "spread": 0.7,
        },
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()


def test_map_embeddings_legacy_dict_with_explicit_algorithm():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{"id": str(i)} for i in range(num_embeddings)]

    # Test with explicit algorithm="umap"
    dataset_identifier = f"unittest-legacy-dict-algo-umap-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        embeddings=embeddings,
        data=data,
        is_public=True,
        projection={"algorithm": "umap", "n_neighbors": 9, "min_dist": 0.3, "n_epochs": 27},
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()

    # Test with explicit algorithm="nomic-project"
    dataset_identifier = f"unittest-legacy-dict-algo-nomic-{gen_temp_identifier()}"
    dataset = atlas.map_data(
        identifier=dataset_identifier,
        id_field="id",
        embeddings=embeddings,
        data=data,
        is_public=True,
        projection={"algorithm": "nomic-project", "n_neighbors": 11, "model": "nomic-project-v2", "n_epochs": 28},
    )
    assert dataset.id is not None
    projection = dataset.maps[0]
    _wait_for_projection_completion(projection)
    dataset.delete()
