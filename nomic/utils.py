import base64
import gc
import random
import sys
from uuid import UUID

import pyarrow as pa

nouns = [
    'newton',
    'einstein',
    'gauss',
    'pascal',
    'laplace',
    'euler',
    'kepler',
    'wiles',
    'neumann',
    'noether',
    'bohr',
    'heisenberg',
    'planck',
    'maxwell',
    'hawking',
    'feynman',
    'curie',
    'poincare',
    'turing',
    'nash',
    'hilbert',
    'godel',
    'galois',
    'ramanujan',
    'lovelace',
    'hypatia',
    'leibniz',
    'copernicus',
    'hubble',
    'darwin',
    'hooke',
    'faraday',
    'pasteur',
    'mendel',
    'galvani',
    'bell',
    'boltzmann',
    'dirac',
    'fermi',
    'oppenheimer',
    'schrodinger',
    'fibonacci',
    'chatelet',
    'herschel',
    'kovalevskaya',
    'somerville',
    'franklin',
    'lepore',
    'yalow',
    'jemison',
    'burnell',
    'bartik',
    'spencer',
    'hamilton',
    'johnson',
    'ride',
    'anning',
    'meitner',
    'hopper',
    'mcclintock',
    'franklin',
    'noether',
    'rubin',
    'wu',
    'fermat',
    'cantor',
    'jordan',
    'riemann',
    'babbage',
    'dijkstra',
    'shannon',
    'erdos',
    'boole',
    'lagrange',
    'lebesgue',
    'legendre',
    'gauss',
    'cauchy',
    'ramanujan',
    'weierstrass',
    'fourier',
    'cavendish',
    'lavoisier',
    'becquerel',
    'avogadro',
    'dalton',
    'rutherford',
    'thomson',
    'bose',
    'fisher',
    'jaynes',
    'bernoulli',
    'alembert',
    'gibbs',
    'pareto',
    'levy',
    'sierpinski',
    'hardy',
    'wigner',
    'mercator',
    'ortelius',
    'speed',
    'schedel',
    'waldseemuller',
    'homann',
    'rennell',
    'ogilby',
    'hondius',
    'moll',
    'blaeu',
    'sanson',
    'borgonio',
    'isle',
    'munster',
    'hollar',
    'kitchin',
    'lecun',
    'bengio',
    'hinton',
    'schmidhuber',
    'fei-fei',
    'koller',
    'russell',
    'norvig',
    'vapnik',
    'avram',
    'bishop',
    'goodfellow',
    'kingma',
    'arora',
    'sutton',
    'breiman',
]

adjectives = [
    'curious',
    'innovative',
    'analytical',
    'creative',
    'logical',
    'adventurous',
    'inquisitive',
    'imaginative',
    'dynamic',
    'quirky',
    'resourceful',
    'persistent',
    'observant',
    'insightful',
    'intuitive',
    'experimental',
    'inspiring',
    'inventive',
    'tenacious',
    'stubborn',
    'lazy',
    'disorganized',
    'inattentive',
    'careless',
    'unpredictable',
    'irrational',
    'impulsive',
    'chaotic',
    'indecisive',
    'cynical',
    'neglectful',
    'overbearing',
    'oblivious',
    'inflexible',
    'lackadaisical',
    'truculent',
    'diffident',
    'lumbering',
]


def arrow_iterator(table: pa.Table):
    # TODO: setting to 10k so it doesn't take too long?
    # Wrote this as a generator so we don't realize the whole table in memory
    reader = table.to_reader(max_chunksize=10_000)
    for batch in reader:
        for item in batch.to_pylist():
            yield item


def b64int(i: int):
    ibytes = int.to_bytes(i, length=8, byteorder='big').lstrip(b'\x00')
    if ibytes == b'':
        ibytes = b'\x00'
    return base64.b64encode(ibytes).decode('utf8').rstrip('=')


def get_random_name():
    return f"{random.choice(adjectives)}-{random.choice(nouns)}"


def assert_valid_project_id(project_id):
    try:
        uuid_obj = UUID(project_id, version=4)
    except ValueError:
        raise ValueError(f"`{project_id}` is not a valid project id.")


def get_object_size_in_bytes(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz
