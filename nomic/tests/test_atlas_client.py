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
import requests

from nomic import AtlasDataset, atlas


def gen_random_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


def test_map_idless_embeddings():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 512)

    dataset = atlas.map_data(identifier=f"unittest-dataset-{random.randint(0,1000)}", embeddings=embeddings)
    AtlasDataset(dataset.identifier).delete()


def test_map_embeddings_with_errors():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)

    name = f'unittest-dataset-{random.randint(0,1000)}'
    # test nested dictionaries
    with pytest.raises(Exception):
        data = [{'key': {'nested_key': 'nested_value'}} for i in range(len(embeddings))]
        dataset = atlas.map_data(embeddings=embeddings, data=data, identifier=name, is_public=True)

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass

    name = f'unittest-dataset-{random.randint(0, 100)}'
    # test underscore
    with pytest.raises(Exception):
        data = [{'__hello': {'hello'}} for i in range(len(embeddings))]
        dataset = atlas.map_data(embeddings=embeddings, data=data, identifier=name, is_public=True)

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass

    name = f'unittest-dataset-{random.randint(0, 100)}'
    # test to long ids
    with pytest.raises(Exception):
        data = [{'id': str(uuid.uuid4()) + 'a'} for i in range(len(embeddings))]
        dataset = atlas.map_data(
            embeddings=embeddings,
            data=data,
            identifier=name,
            id_field='id',
            is_public=True,
        )

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass


def test_map_text_errors():
    # no indexed field
    name = f'unittest-dataset-{random.randint(0, 100)}'
    with pytest.raises(Exception):
        dataset = atlas.map_data(
            data=[{'key': 'a'}],
            indexed_field='text',
            is_public=True,
            identifier=name,
            description='test map description',
        )

    try:
        AtlasDataset(name).delete()
    except BaseException:
        pass


def test_date_metadata():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [
        {'my_date': datetime(2022, 1, i), 'my_random_date': gen_random_datetime()}
        for i in range(1, len(embeddings) + 1)
    ]

    dataset = atlas.map_data(
        embeddings=embeddings, identifier=f"unittest-dataset-{random.randint(0,1000)}", data=data, is_public=True
    )

    assert dataset.id

    dataset.delete()

    # put an invalid iso timestamp after the first valid isotimestamp , make sure the client fails
    with pytest.raises(Exception):
        data[1]['my_date'] = data[1]['my_date'] + 'asdf'
        dataset = atlas.map_data(
            embeddings=embeddings,
            identifier=f"unittest-dataset-{random.randint(0,1000)}",
            id_field='id',
            data=data,
            is_public=True,
        )


def test_dataset_with_updates():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0} for i in range(len(embeddings))]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier='test_map_embedding_progressive',
        id_field='id',
        data=data,
        is_public=True,
        topic_model=dict(build_topic_model=False),
    )

    embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0} for i in range(len(embeddings))]

    current_dataset = AtlasDataset(dataset.identifier)

    with current_dataset.wait_for_dataset_lock():
        current_dataset.add_data(data=data, embeddings=embeddings)
        current_dataset.update_indices()

    with current_dataset.wait_for_dataset_lock():
        current_dataset.delete()


def test_topics():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ['foo', 'bar', 'baz', 'bat']
    dates = [datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1)]
    data = [
        {'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0, 'text': texts[i % 4], 'date': dates[i % 3]}
        for i in range(len(embeddings))
    ]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{random.randint(0,1000)}",
        id_field='id',
        data=data,
        is_public=True,
        topic_model=dict(topic_label_field='text'),
    )

    with dataset.wait_for_dataset_lock():
        time.sleep(5)
        assert len(dataset.maps[0].topics.metadata) > 0

        q = np.random.random((3, 10))
        assert len(dataset.maps[0].topics.vector_search_topics(q, depth=1, k=3)['topics']) == 3
        assert isinstance(dataset.maps[0].topics.group_by_topic(topic_depth=1), list)

        # start = datetime(2019, 1, 1)
        # end = datetime(2025, 1, 1)
        # assert isinstance(dataset.maps[0].topics.get_topic_density("date", start, end), dict)

        dataset.delete()


def test_data():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ['foo', 'bar', 'baz', 'bat']
    data = [
        {'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0, 'text': str(i)}
        for i in range(len(embeddings))
    ]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{random.randint(0,1000)}",
        id_field='id',
        data=data,
        is_public=True,
        topic_model=dict(build_topic_model=True, community_description_target_field='text'),
    )

    with dataset.wait_for_dataset_lock():
        df = dataset.maps[0].data.df
        assert len(df) > 0
        assert "text" in df.columns
        dataset.delete()


words = [
    'foo',
    'bar',
    'baz',
    'bat',
    'glorp',
    'gloop',
    'glib',
    'glub',
    'florp',
    'floop',
    'flib',
    'flub',
    'blorp',
    'bloop',
    'blib',
    'blub',
    'slorp',
    'sloop',
    'slib',
    'slub',
    'clorp',
    'cloop',
    'clib',
    'club',
    'plorp',
    'ploop',
    'plib',
    'plub',
    'zlorp',
    'zloop',
    'zlib',
    'zlub',
    'xlorp',
    'xloop',
    'xlib',
    'xlub',
    'vlorp',
    'vloop',
    'vlib',
    'vlub',
    'nlorp',
    'nloop',
    'nlib',
    'nlub',
    'mlorp',
    'mloop',
    'mlib',
    'mlub',
]


def test_flawed_ids():
    """
    Check that null and empty strings do not block an index build.
    """
    p = AtlasDataset(f"unittest-dataset-{random.randint(0,1000)}", unique_id_field='id')

    elements = []
    for i in range(10):
        if i % 3 == 0 and i % 5 == 0:
            elements.append({'text': 'fizzbuzz', 'id': str(i)})
        elif i % 3 == 0:
            elements.append({'text': 'fizz', 'id': str(i)})
        elif i % 5 == 0:
            elements.append({'text': 'buzz', 'id': str(i)})
    p.add_data(data=elements)
    with pytest.raises(ValueError):
        p.add_data(data=[{'text': 'fizzbuzz', 'id': None}])
    with pytest.raises(ValueError):
        p.add_data(data=[{'text': 'fizzbuzz', 'id': 'A' * 100}])
    p.delete()


def test_weird_inputs():
    """
    Check that null and empty strings do not block an index build.
    """
    dataset = AtlasDataset(f"unittest-dataset-{random.randint(0,1000)}", unique_id_field='id')

    elements = []
    for i in range(20):
        if i % 3 == 0 and i % 5 == 0:
            elements.append({'text': 'fizzbuzz', 'id': str(i)})
        elif i % 3 == 0:
            elements.append({'text': 'fizz', 'id': str(i)})
        elif i % 5 == 0:
            elements.append({'text': 'buzz', 'id': str(i)})
        elif i % 7 == 0:
            elements.append({'text': None, 'id': str(i)})
        elif i % 2 == 0:
            elements.append({'text': '', 'id': str(i)})
        else:
            elements.append({'text': 'foo', 'id': str(i)})
    dataset.add_data(data=elements)
    dataset.create_index(indexed_field='text', topic_model=True)
    with dataset.wait_for_dataset_lock():
        assert True
    dataset.delete()


def test_map_embeddings():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4())} for i in range(len(embeddings))]

    dataset = atlas.map_data(
        embeddings=embeddings,
        identifier=f"unittest-dataset-{random.randint(0,1000)}",
        id_field='id',
        data=data,
        is_public=True,
    )

    map = dataset.maps[0]

    num_tries = 0
    while map.project.is_locked:
        time.sleep(10)
        num_tries += 1
        if num_tries > 5:
            raise TimeoutError('Timed out while waiting for project to unlock')

    retrieved_embeddings = map.embeddings.latent

    assert dataset.total_datums == num_embeddings
    assert retrieved_embeddings.shape[0] == num_embeddings
    map = dataset.maps[0]

    assert len(map.topics.df) == 20

    assert isinstance(map.topics.hierarchy, dict)

    dataset.create_index()
    with dataset.wait_for_dataset_lock():
        neighbors, _ = map.embeddings.vector_search(queries=np.random.rand(1, 10), k=2)
        assert len(neighbors[0]) == 2

    for map in dataset.projections:
        assert map.map_link

    map.tags.add(ids=[data[0]['id']], tags=['my_tag'])

    assert len(map.tags.get_tags()['my_tag']) == 1

    map.tags.remove(ids=[data[0]['id']], tags=['my_tag'])

    assert 'my_tag' not in map.tags.get_tags()

    dataset.delete()


def test_map_text_pandas():
    size = 50
    data = pd.DataFrame(
        {
            'field': [str(uuid.uuid4()) for i in range(size)],
            'color': [random.choice(['red', 'blue', 'green']) for i in range(size)],
        }
    )

    dataset = atlas.map_data(identifier='UNITTEST_pandas_text', indexed_field="color", data=data, is_public=True)

    assert dataset.total_datums == 50

    dataset.delete()


def test_map_text_arrow():
    size = 50
    data = pa.Table.from_pydict(
        {
            'field': [str(uuid.uuid4()) for i in range(size)],
            'id': [str(uuid.uuid4()) for i in range(size)],
            'color': [random.choice(['red', 'blue', 'green']) for i in range(size)],
        }
    )

    dataset = atlas.map_data(
        identifier='UNITTEST_arrow_text',
        id_field='id',
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
