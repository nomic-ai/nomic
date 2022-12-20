import uuid
import time
import tempfile
import datetime

from nomic import AtlasClient
import pytest
import random
import time
import numpy as np

def test_map_embeddings_with_errors():
    atlas = AtlasClient()

    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)


    #test nested dictionaries
    with pytest.raises(Exception):
        data = [{'hello': {'hello'}} for i in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST1',
                                        is_public=True,
                                        reset_project_if_exists=True)

    #test underscore
    with pytest.raises(Exception):
        data = [{'__hello': {'hello'} } for i in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST1',
                                        is_public=True,
                                        reset_project_if_exists=True)

    #test non-matching keys across metadatums
    with pytest.raises(Exception):
        data = [{'hello': 'a'} for i in range(len(embeddings))]
        data[1]['goodbye'] = 'b'
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST1',
                                        is_public=True,
                                        reset_project_if_exists=True)

    #test to long ids
    with pytest.raises(Exception):
        data = [{'id': str(uuid.uuid4())+'a'} for i in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST1',
                                        id_field='id',
                                        is_public=True,
                                        reset_project_if_exists=True)

    #test duplicate keys error
    with pytest.raises(Exception):
        data = [{'b': 'a'} for i in range(len(embeddings))]
        data[1]['goodbye'] = 'b'
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST1',
                                        is_public=True,
                                        reset_project_if_exists=True)

    #fail on to large metadata
    with pytest.raises(Exception):
        embeddings = np.random.rand(1000, 10)
        data = [{'string': ''.join(['a'] * (1048576 // 10))} for _ in range(len(embeddings))]
        response = atlas.map_embeddings(embeddings=embeddings,
                                        data=data,
                                        map_name='UNITTEST1',
                                        is_public=True,
                                        reset_project_if_exists=True)


def test_map_embeddings():
    atlas = AtlasClient()

    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4())} for i in range(len(embeddings))]

    response = atlas.map_embeddings(embeddings=embeddings,
                                    map_name='UNITTEST1',
                                    id_field='id',
                                    data=data,
                                    is_public=True,
                                    reset_project_if_exists=True)

    assert response['project_id']
    project_id = response['project_id']

    project = atlas.get_project('UNITTEST1')
    project = atlas._get_project_by_id(project_id=project['id'])
    atlas_index_id = project['atlas_indices'][0]['id']

    time.sleep(60)
    with tempfile.TemporaryDirectory() as td:
        atlas.download_embeddings(project_id, atlas_index_id, td)

    atlas.delete_project(project_id=response['project_id'])


def test_date_metadata():
    atlas = AtlasClient()

    num_embeddings = 10
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'my_date': datetime.datetime(2022, 1, i).isoformat()} for i in range(1, len(embeddings)+1)]

    response = atlas.map_embeddings(embeddings=embeddings,
                                    map_name='UNITTEST1',
                                    data=data,
                                    is_public=True,
                                    reset_project_if_exists=True)

    assert response['project_id']

    atlas.delete_project(project_id=response['project_id'])


    #put an invalid iso timestamp after the first valid isotimestamp , make sure the client fails
    with pytest.raises(Exception):
        data[1]['my_date'] = data[1]['my_date']+'asdf'
        response = atlas.map_embeddings(embeddings=embeddings,
                                        map_name='UNITTEST1',
                                        id_field='id',
                                        data=data,
                                        is_public=True,
                                        reset_project_if_exists=True)



def test_map_text_errors():
    atlas = AtlasClient()

    # no indexed field
    with pytest.raises(Exception):
        response = atlas.map_text(data=[{'key': 'a'}],
                                  indexed_field='text',
                                  is_public=True,
                                  map_name='UNITTEST1',
                                  map_description='test map description',
                                  num_workers=1,
                                  reset_project_if_exists=True)


def test_map_embedding_progressive():
    atlas = AtlasClient()

    num_embeddings = 1000
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0} for i in range(len(embeddings))]

    response = atlas.map_embeddings(embeddings=embeddings,
                                    map_name='UNITTEST1',
                                    id_field='id',
                                    data=data,
                                    is_public=True,
                                    build_topic_model=False,
                                    reset_project_if_exists=True)

    embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0} for i in range(len(embeddings))]

    current_project = atlas.get_project(response['project_name'])

    while True:
        time.sleep(10)
        if atlas.is_project_accepting_data(project_id=current_project['id']):
            response = atlas.map_embeddings(embeddings=embeddings,
                                            map_name=current_project['project_name'],
                                            colorable_fields=['upload'],
                                            id_field='id',
                                            data=data,
                                            build_topic_model=False,
                                            is_public=True,
                                            add_datums_if_exists=True
                                            )
            return True


