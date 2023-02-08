import datetime
import random
import tempfile
import time
import uuid

import numpy as np
import pytest
import requests

from nomic import AtlasProject, atlas


# def test_map_embeddings_with_errors():
#
#     num_embeddings = 10
#     embeddings = np.random.rand(num_embeddings, 10)
#
#     # test nested dictionaries
#     with pytest.raises(Exception):
#         data = [{'hello': {'hello'}} for i in range(len(embeddings))]
#         response = atlas.map_embeddings(
#             embeddings=embeddings, data=data, map_name='UNITTEST1', is_public=True, reset_project_if_exists=True
#         )
#
#     # test underscore
#     with pytest.raises(Exception):
#         data = [{'__hello': {'hello'}} for i in range(len(embeddings))]
#         response = atlas.map_embeddings(
#             embeddings=embeddings, data=data, map_name='UNITTEST1', is_public=True, reset_project_if_exists=True
#         )
#
#     # test non-matching keys across metadatums
#     with pytest.raises(Exception):
#         data = [{'hello': 'a'} for i in range(len(embeddings))]
#         data[1]['goodbye'] = 'b'
#         response = atlas.map_embeddings(
#             embeddings=embeddings, data=data, map_name='UNITTEST1', is_public=True, reset_project_if_exists=True
#         )
#
#     # test to long ids
#     with pytest.raises(Exception):
#         data = [{'id': str(uuid.uuid4()) + 'a'} for i in range(len(embeddings))]
#         response = atlas.map_embeddings(
#             embeddings=embeddings,
#             data=data,
#             map_name='UNITTEST1',
#             id_field='id',
#             is_public=True,
#             reset_project_if_exists=True,
#         )
#
#     # test duplicate keys error
#     with pytest.raises(Exception):
#         data = [{'b': 'a'} for i in range(len(embeddings))]
#         data[1]['goodbye'] = 'b'
#         response = atlas.map_embeddings(
#             embeddings=embeddings, data=data, map_name='UNITTEST1', is_public=True, reset_project_if_exists=True
#         )
#
#     # fail on to large metadata
#     with pytest.raises(Exception):
#         embeddings = np.random.rand(1000, 10)
#         data = [{'string': ''.join(['a'] * (1048576 // 10))} for _ in range(len(embeddings))]
#         response = atlas.map_embeddings(
#             embeddings=embeddings, data=data, map_name='UNITTEST1', is_public=True, reset_project_if_exists=True
#         )


def test_map_embeddings():
    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4())} for i in range(len(embeddings))]

    project = atlas.map_embeddings(
        embeddings=embeddings,
        map_name='UNITTEST1',
        id_field='id',
        data=data,
        is_public=True,
        shard_size=5,
        reset_project_if_exists=True,
    )

    map = project.get_map(name='UNITTEST1')

    time.sleep(10)
    with tempfile.TemporaryDirectory() as td:
        retrieved_embeddings = map.download_embeddings(td)

    assert project.total_datums == num_embeddings

    project = AtlasProject(name='UNITTEST1')
    map = project.get_map(name='UNITTEST1')

    project.create_index(index_name='My new index')

    # neighbors = map.get_nearest_neighbors(queries=np.random.rand(1, 10), k=2)
    # print(neighbors)

    for map in project.projections:
        assert map.map_link

    project.delete()


def test_date_metadata():
    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'my_date': datetime.datetime(2022, 1, i).isoformat()} for i in range(1, len(embeddings) + 1)]

    project = atlas.map_embeddings(
        embeddings=embeddings, map_name='UNITTEST1', data=data, is_public=True, reset_project_if_exists=True
    )

    assert project.id

    project.delete()

    # put an invalid iso timestamp after the first valid isotimestamp , make sure the client fails
    with pytest.raises(Exception):
        data[1]['my_date'] = data[1]['my_date'] + 'asdf'
        project = atlas.map_embeddings(
            embeddings=embeddings,
            map_name='UNITTEST1',
            id_field='id',
            data=data,
            is_public=True,
            reset_project_if_exists=True,
        )


def test_map_text_errors():

    # no indexed field
    with pytest.raises(Exception):
        project = atlas.map_text(
            data=[{'key': 'a'}],
            indexed_field='text',
            is_public=True,
            map_name='UNITTEST1',
            map_description='test map description',
            num_workers=1,
            reset_project_if_exists=True,
        )


def test_map_embedding_progressive():

    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0} for i in range(len(embeddings))]

    project = atlas.map_embeddings(
        embeddings=embeddings,
        map_name='UNITTEST1',
        id_field='id',
        data=data,
        is_public=True,
        build_topic_model=False,
        reset_project_if_exists=True,
    )

    embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0} for i in range(len(embeddings))]

    current_project = AtlasProject(name=project.name)

    while True:
        time.sleep(10)
        if project.is_accepting_data:
            project = atlas.map_embeddings(
                embeddings=embeddings,
                map_name=current_project.name,
                colorable_fields=['upload'],
                id_field='id',
                data=data,
                build_topic_model=False,
                is_public=True,
                add_datums_if_exists=True,
            )
            project.delete()
            return True
