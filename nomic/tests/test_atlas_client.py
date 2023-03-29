import datetime
import random
import tempfile
import time
import uuid
from datetime import datetime, timedelta
import numpy as np
import pytest
import requests
from nomic import AtlasProject, atlas


def gen_random_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


def test_map_idless_embeddings():
    num_embeddings = 50
    embeddings = np.random.rand(num_embeddings, 512)

    response = atlas.map_embeddings(embeddings=embeddings)
    print(response)


def test_map_embeddings_with_errors():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)

    # test nested dictionaries
    with pytest.raises(Exception):
        data = [{'key': {'nested_key': 'nested_value'}} for i in range(len(embeddings))]
        response = atlas.map_embeddings(
            embeddings=embeddings, data=data, name='UNITTEST1', is_public=True, reset_project_if_exists=True
        )

    # test underscore
    with pytest.raises(Exception):
        data = [{'__hello': {'hello'}} for i in range(len(embeddings))]
        response = atlas.map_embeddings(
            embeddings=embeddings, data=data, name='UNITTEST1', is_public=True, reset_project_if_exists=True
        )

    # test to long ids
    with pytest.raises(Exception):
        data = [{'id': str(uuid.uuid4()) + 'a'} for i in range(len(embeddings))]
        response = atlas.map_embeddings(
            embeddings=embeddings,
            data=data,
            name='UNITTEST1',
            id_field='id',
            is_public=True,
            reset_project_if_exists=True,
        )


def test_map_embeddings():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4())} for i in range(len(embeddings))]

    project = atlas.map_embeddings(
        embeddings=embeddings,
        name='UNITTEST1',
        id_field='id',
        data=data,
        is_public=True,
        reset_project_if_exists=True,
    )

    map = project.get_map(name='UNITTEST1')

    time.sleep(10)
    with tempfile.TemporaryDirectory() as td:
        retrieved_embeddings = map.download_embeddings(td)

    assert project.total_datums == num_embeddings

    project = AtlasProject(name='UNITTEST1')
    map = project.get_map(name='UNITTEST1')

    project.create_index(name='My new index')
    with project.wait_for_project_lock():
        neighbors, _ = map.vector_search(queries=np.random.rand(1, 10), k=2)
        assert len(neighbors[0]) == 2

    for map in project.projections:
        assert map.map_link

    map.tag(ids=[data[0]['id']], tags=['my_tag'])

    assert len(map.get_tags()['my_tag']) == 1

    map.remove_tags(ids=[data[0]['id']], tags=['my_tag'])

    assert 'my_tag' not in map.get_tags()

    project.delete()


def test_date_metadata():
    num_embeddings = 20
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'my_date': datetime.datetime(2022, 1, i),
             'my_random_date': gen_random_datetime()} for i in range(1, len(embeddings) + 1)]

    project = atlas.map_embeddings(
        embeddings=embeddings, name='test_date_metadata', data=data, is_public=True, reset_project_if_exists=True
    )

    assert project.id

    project.delete()

    # put an invalid iso timestamp after the first valid isotimestamp , make sure the client fails
    with pytest.raises(Exception):
        data[1]['my_date'] = data[1]['my_date'] + 'asdf'
        project = atlas.map_embeddings(
            embeddings=embeddings,
            name='UNITTEST1',
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
            name='test_map_text_errors',
            description='test map description',
            reset_project_if_exists=True,
        )


def test_map_embedding_progressive():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0} for i in range(len(embeddings))]

    project = atlas.map_embeddings(
        embeddings=embeddings,
        name='test_map_embedding_progressive',
        id_field='id',
        data=data,
        is_public=True,
        build_topic_model=False,
        reset_project_if_exists=True,
    )

    embeddings = np.random.rand(num_embeddings, 10) + np.ones(shape=(num_embeddings, 10))
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0} for i in range(len(embeddings))]

    current_project = AtlasProject(name=project.name)

    with current_project.wait_for_project_lock():
        project = atlas.map_embeddings(
            embeddings=embeddings,
            name=current_project.name,
            colorable_fields=['upload'],
            id_field='id',
            data=data,
            build_topic_model=False,
            is_public=True,
            add_datums_if_exists=True,
        )
    with pytest.raises(Exception):
        # Try adding a bad field.
        with current_project.wait_for_project_lock():
            data = [{'invalid_field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 1.0} for i in
                    range(len(embeddings))]

            current_project = AtlasProject(name=project.name)

            with current_project.wait_for_project_lock():
                project = atlas.map_embeddings(
                    embeddings=embeddings,
                    name=current_project.name,
                    colorable_fields=['upload'],
                    id_field='id',
                    data=data,
                    build_topic_model=False,
                    is_public=True,
                    add_datums_if_exists=True,
                )

    current_project.delete()


def test_topics():
    num_embeddings = 100
    embeddings = np.random.rand(num_embeddings, 10)
    texts = ['foo', 'bar', 'baz', 'bat']
    data = [{'field': str(uuid.uuid4()), 'id': str(uuid.uuid4()), 'upload': 0.0, 'text': texts[i % 4]}
            for i in range(len(embeddings))]

    p = atlas.map_embeddings(
        embeddings=embeddings,
        name='test_topics',
        id_field='id',
        data=data,
        is_public=True,
        build_topic_model=True,
        topic_label_field='text',
        reset_project_if_exists=True,
    )

    with p.wait_for_project_lock():
        assert len(p.maps[0].get_topic_data()) > 0

        q = np.random.random((3, 10))
        assert len(p.maps[0].vector_search_topics(q, depth=1, k=3)['topics']) == 3
        p.delete()


words = [
    'foo', 'bar', 'baz', 'bat',
    'glorp', 'gloop', 'glib', 'glub',
    'florp', 'floop', 'flib', 'flub',
    'blorp', 'bloop', 'blib', 'blub',
    'slorp', 'sloop', 'slib', 'slub',
    'clorp', 'cloop', 'clib', 'club',
    'plorp', 'ploop', 'plib', 'plub',
    'zlorp', 'zloop', 'zlib', 'zlub',
    'xlorp', 'xloop', 'xlib', 'xlub',
    'vlorp', 'vloop', 'vlib', 'vlub',
    'nlorp', 'nloop', 'nlib', 'nlub',
    'mlorp', 'mloop', 'mlib', 'mlub'
]


def test_interactive_workflow():
    p = AtlasProject(name='UNITTEST1',
                     modality='text',
                     unique_id_field='id',
                     reset_project_if_exists=True
                     )

    p.add_text(data=[{'text': random.choice(words), 'id': i} for i in range(100)])

    p.create_index(name='UNITTEST1',
                   indexed_field='text',
                   build_topic_model=True
                   )

    assert p.total_datums == 100

    # Test ability to add more data to a project and have the ids coerced.
    with p.wait_for_project_lock():
        p.add_text(data=[{'text': random.choice(words), 'id': i} for i in range(100, 200)])
        p.create_index(name='UNITTEST1',
                       indexed_field='text',
                       build_topic_model=True
                       )
        assert p.total_datums == 200

    with p.wait_for_project_lock():
        p.delete()


def test_weird_inputs():
    """
    Check that null and empty strings do not block an index build.
    """
    p = AtlasProject(
        name='test_weird_inputs',
        modality='text',
        unique_id_field='id',
        reset_project_if_exists=True
    )

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
    p.create_index(
        name='test_weird_inputs',
        indexed_field='text',
        build_topic_model=True
    )
    with p.wait_for_project_lock():
        assert True