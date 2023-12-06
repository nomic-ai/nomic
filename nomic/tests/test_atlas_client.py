import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timedelta
import numpy as np
import pytest
import requests
from nomic import AtlasDataset, atlas
import pyarrow as pa
import pandas as pd

def gen_random_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


def test_map_idless_embeddings():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 512)

    dataset = atlas.map_data(name="test1", embeddings=embeddings)

    AtlasDataset(identifier=dataset.identifier).delete()


def test_map_embeddings_with_errors():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)

    # test nested dictionaries
    with pytest.raises(Exception):
        data = [{'key': {'nested_key': 'nested_value'}} for i in range(len(embeddings))]
        response = atlas.map_data(
            embeddings=embeddings, data=data, name='UNITTEST1', is_public=True
        )

    # test underscore
    with pytest.raises(Exception):
        data = [{'__hello': {'hello'}} for i in range(len(embeddings))]
        response = atlas.map_data(
            embeddings=embeddings, data=data, name='UNITTEST1', is_public=True
        )

    # test to long ids
    with pytest.raises(Exception):
        data = [{'id': str(uuid.uuid4()) + 'a'} for i in range(len(embeddings))]
        response = atlas.map_data(
            embeddings=embeddings,
            data=data,
            name='UNITTEST1',
            id_field='id',
            is_public=True,
        )


def test_map_text_errors():
    # no indexed field
    with pytest.raises(Exception):
        project = atlas.map_data(
            data=[{'key': 'a'}],
            indexed_field='text',
            is_public=True,
            name='test_map_text_errors',
            description='test map description',
        )


def test_date_metadata():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [
        {'my_date': datetime(2022, 1, i), 'my_random_date': gen_random_datetime()}
        for i in range(1, len(embeddings) + 1)
    ]

    project = atlas.map_data(
        embeddings=embeddings, name='test_date_metadata', data=data, is_public=True
    )

    assert project.id

    project.delete()

    # put an invalid iso timestamp after the first valid isotimestamp , make sure the client fails
    with pytest.raises(Exception):
        data[1]['my_date'] = data[1]['my_date'] + 'asdf'
        project = atlas.map_data(
            embeddings=embeddings,
            name='UNITTEST1',
            id_field='id',
            data=data,
            is_public=True,
        )


# def test_map_embedding_progressive():
#     num_embeddings = 100
#     embeddings = np.random.rand(num_embeddings, 10)
#     data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0} for i in range(len(embeddings))]
#
#     project = atlas.map_data(
#         embeddings=embeddings,
#         name='test_map_embedding_progressive',
#         id_field='id',
#         data=data,
#         is_public=True,
#         topic_model=dict(build_topic_model=False)
#     )
#
#     embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
#     data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0} for i in range(len(embeddings))]
#
#     current_project = AtlasDataset(project.name)
#
#     with current_project.wait_for_dataset_lock():
#         project = atlas.map_data(
#             embeddings=embeddings,
#             name=current_project.name,
#             colorable_fields=['upload'],
#             id_field='id',
#             data=data,
#             topic_model=dict(build_topic_model=False),
#             is_public=True,
#         )
#     with pytest.raises(Exception):
#         # Try adding a bad field.
#         with current_project.wait_for_dataset_lock():
#             data = [
#                 {'invalid_field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0}
#                 for i in range(len(embeddings))
#             ]
#
#             current_project = AtlasDataset(project.name)
#
#             with current_project.wait_for_dataset_lock():
#                 project = atlas.map_data(
#                     embeddings=embeddings,
#                     name=current_project.name,
#                     colorable_fields=['upload'],
#                     id_field='id',
#                     data=data,
#                     topic_model=dict(build_topic_model=False),
#                     is_public=True,
#                 )
#
#     current_project.delete()


def test_topics():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ['foo', 'bar', 'baz', 'bat']
    dates = [datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1)]
    data = [
        {'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0, 'text': texts[i % 4], 'date': dates[i % 3]}
        for i in range(len(embeddings))
    ]

    project = atlas.map_data(
        embeddings=embeddings,
        name='test_topics',
        id_field='id',
        data=data,
        is_public=True,
        topic_model=dict(build_topic_model=True, community_description_target_field='text'),
    )

    with project.wait_for_dataset_lock():
        assert len(project.maps[0].topics.metadata) > 0

        q = np.random.random((3, 10))
        assert len(project.maps[0].topics.vector_search_topics(q, depth=1, k=3)['topics']) == 3
        assert isinstance(project.maps[0].topics.group_by_topic(topic_depth=1), list)

        start = datetime(2019, 1, 1)
        end = datetime(2025, 1, 1)
        assert isinstance(project.maps[0].topics.get_topic_density("date", start, end), dict)

        project.delete()

def test_data():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ['foo', 'bar', 'baz', 'bat']
    data = [
        {'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0, 'text': str(i)}
        for i in range(len(embeddings))
    ]

    project = atlas.map_data(
        embeddings=embeddings,
        name='test_topics',
        id_field='id',
        data=data,
        is_public=True,
        topic_model=dict(build_topic_model=True, community_description_target_field='text'),
    )

    with project.wait_for_dataset_lock():
        df = project.maps[0].data.df
        assert len(df) > 0
        assert "text" in df.columns
        project.delete()

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


def test_interactive_workflow():
    p = AtlasDataset('UNITTEST1', modality='text', unique_id_field='id')

    p.add_text(data=[{'text': random.choice(words), 'id': i} for i in range(100)])

    p.create_index(name='UNITTEST1', indexed_field='text', build_topic_model=True)

    assert p.total_datums == 100

    # Test ability to add more data to a project and have the ids coerced.
    with p.wait_for_dataset_lock():
        p.add_text(data=[{'text': random.choice(words), 'id': i} for i in range(100, 200)])
        p.create_index(name='UNITTEST1', indexed_field='text', build_topic_model=True)
        assert p.total_datums == 200

    with p.wait_for_dataset_lock():
        p.delete()


def test_flawed_ids():
    """
    Check that null and empty strings do not block an index build.
    """
    p = AtlasDataset('test_flawed_ids', modality='text', unique_id_field='id')

    elements = []
    for i in range(10):
        if i % 3 == 0 and i % 5 == 0:
            elements.append({'text': 'fizzbuzz', 'id': str(i)})
        elif i % 3 == 0:
            elements.append({'text': 'fizz', 'id': str(i)})
        elif i % 5 == 0:
            elements.append({'text': 'buzz', 'id': str(i)})
    p.add_text(data=elements)
    with pytest.raises(ValueError):
        p.add_text(data=[{'text': 'fizzbuzz', 'id': None}])
    with pytest.raises(ValueError):
        p.add_text(data=[{'text': 'fizzbuzz', 'id': 'A' * 100}])
    p.delete()


def test_weird_inputs():
    """
    Check that null and empty strings do not block an index build.
    """
    p = AtlasDataset('test_weird_inputs', modality='text', unique_id_field='id')

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
    p.add_text(data=elements)
    p.create_index(name='test_weird_inputs', indexed_field='text', build_topic_model=True)
    with p.wait_for_dataset_lock():
        assert True
    p.delete()


def test_map_embeddings():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4())} for i in range(len(embeddings))]

    project = atlas.map_data(
        embeddings=embeddings,
        name='UNITTEST1',
        id_field='id',
        data=data,
        is_public=True,
    )

    map = project.get_map(name='UNITTEST1')

    num_tries = 0
    while map.project.is_locked:
        time.sleep(10)
        num_tries += 1
        if num_tries > 5:
            raise TimeoutError('Timed out while waiting for project to unlock')

    retrieved_embeddings = map.embeddings.latent

    assert project.total_datums == num_embeddings
    assert retrieved_embeddings.shape[0] == num_embeddings

    project = AtlasDataset('UNITTEST1')
    map = project.get_map(name='UNITTEST1')

    assert len(map.topics.df) == 20

    assert isinstance(map.topics.hierarchy, dict)

    project.create_index(name='My new index')
    with project.wait_for_dataset_lock():
        neighbors, _ = map.embeddings.vector_search(queries=np.random.rand(1, 10), k=2)
        assert len(neighbors[0]) == 2

    for map in project.projections:
        assert map.map_link

    map.tags.add(ids=[data[0]['id']], tags=['my_tag'])

    assert len(map.tags.get_tags()['my_tag']) == 1

    map.tags.remove(ids=[data[0]['id']], tags=['my_tag'])

    assert 'my_tag' not in map.tags.get_tags()

    project.delete()


def test_map_text_pandas():
    size = 50
    data = pd.DataFrame({
        'field': [str(uuid.uuid4()) for i in range(size)],
        'id': [str(uuid.uuid4()) for i in range(size)],
        'color': [random.choice(['red', 'blue', 'green']) for i in range(size)],
    })

    project = atlas.map_data(
        name='UNITTEST_pandas_text',
        id_field='id',
        indexed_field="color",
        data=data,
        is_public=True,
        colorable_fields=['color'],
    )

    map = project.get_map(name='UNITTEST_pandas_text')

    assert project.total_datums == 50

    project.delete()

    
def test_map_text_arrow():
    size = 50
    data = pa.Table.from_pydict({
        'field': [str(uuid.uuid4()) for i in range(size)],
        'id': [str(uuid.uuid4()) for i in range(size)],
        'color': [random.choice(['red', 'blue', 'green']) for i in range(size)],
    })

    project = atlas.map_data(
        name='UNITTEST_arrow_text',
        id_field='id',
        indexed_field="color",
        data=data,
        is_public=True,
        colorable_fields=['color'],
    )

    map = project.get_map(name='UNITTEST_arrow_text')

    assert project.total_datums == 50

    project.delete()


# def test_map_text_iterator():
#     size = 50
#     data = [
#         {
#             'field': str(uuid.uuid4()),
#             'id': str(uuid.uuid4()),
#             'color': random.choice(['red', 'blue', 'green'])
#         }
#         for _ in range(size)
#     ]
#
#     data_iter = iter(data)
#
#     project = atlas.map_data(
#         name='UNITTEST_pandas_text',
#         id_field='id',
#         indexed_field="color",
#         data=data_iter,
#         is_public=True,
#         colorable_fields=['color'],
#         reset_project_if_exists=True,
#     )
#
#     map = project.get_map(name='UNITTEST_pandas_text')
#     assert project.total_datums == 50
#     project.delete()
