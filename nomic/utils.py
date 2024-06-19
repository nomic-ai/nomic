import base64
import gc
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

import pyarrow as pa
import requests
from pyarrow import ipc

nouns = [
    "newton",
    "einstein",
    "gauss",
    "pascal",
    "laplace",
    "euler",
    "kepler",
    "wiles",
    "neumann",
    "noether",
    "bohr",
    "heisenberg",
    "planck",
    "maxwell",
    "hawking",
    "feynman",
    "curie",
    "poincare",
    "turing",
    "nash",
    "hilbert",
    "godel",
    "galois",
    "ramanujan",
    "lovelace",
    "hypatia",
    "leibniz",
    "copernicus",
    "hubble",
    "darwin",
    "hooke",
    "faraday",
    "pasteur",
    "mendel",
    "galvani",
    "bell",
    "boltzmann",
    "dirac",
    "fermi",
    "oppenheimer",
    "schrodinger",
    "fibonacci",
    "chatelet",
    "herschel",
    "kovalevskaya",
    "somerville",
    "franklin",
    "lepore",
    "yalow",
    "jemison",
    "burnell",
    "bartik",
    "spencer",
    "hamilton",
    "johnson",
    "ride",
    "anning",
    "meitner",
    "hopper",
    "mcclintock",
    "franklin",
    "noether",
    "rubin",
    "wu",
    "fermat",
    "cantor",
    "jordan",
    "riemann",
    "babbage",
    "dijkstra",
    "shannon",
    "erdos",
    "boole",
    "lagrange",
    "lebesgue",
    "legendre",
    "gauss",
    "cauchy",
    "ramanujan",
    "weierstrass",
    "fourier",
    "cavendish",
    "lavoisier",
    "becquerel",
    "avogadro",
    "dalton",
    "rutherford",
    "thomson",
    "bose",
    "fisher",
    "jaynes",
    "bernoulli",
    "alembert",
    "gibbs",
    "pareto",
    "levy",
    "sierpinski",
    "hardy",
    "wigner",
    "mercator",
    "ortelius",
    "speed",
    "schedel",
    "waldseemuller",
    "homann",
    "rennell",
    "ogilby",
    "hondius",
    "moll",
    "blaeu",
    "sanson",
    "borgonio",
    "isle",
    "munster",
    "hollar",
    "kitchin",
    "lecun",
    "bengio",
    "hinton",
    "schmidhuber",
    "fei-fei",
    "koller",
    "russell",
    "norvig",
    "vapnik",
    "avram",
    "bishop",
    "goodfellow",
    "kingma",
    "arora",
    "sutton",
    "breiman",
]

adjectives = [
    "curious",
    "innovative",
    "analytical",
    "creative",
    "logical",
    "adventurous",
    "inquisitive",
    "imaginative",
    "dynamic",
    "quirky",
    "resourceful",
    "persistent",
    "observant",
    "insightful",
    "intuitive",
    "experimental",
    "inspiring",
    "inventive",
    "tenacious",
    "stubborn",
    "lazy",
    "disorganized",
    "inattentive",
    "careless",
    "unpredictable",
    "irrational",
    "impulsive",
    "chaotic",
    "indecisive",
    "cynical",
    "neglectful",
    "overbearing",
    "oblivious",
    "inflexible",
    "lackadaisical",
    "truculent",
    "diffident",
    "lumbering",
]


def arrow_iterator(table: pa.Table):
    # TODO: setting to 10k so it doesn't take too long?
    # Wrote this as a generator so we don't realize the whole table in memory
    reader = table.to_reader(max_chunksize=10_000)
    for batch in reader:
        for item in batch.to_pylist():
            yield item


def b64int(i: int):
    ibytes = int.to_bytes(i, length=8, byteorder="big").lstrip(b"\x00")
    if ibytes == b"":
        ibytes = b"\x00"
    return base64.b64encode(ibytes).decode("utf8").rstrip("=")


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


# Helpful function for downloading feather files
# Best for small feather files
def download_feather(
    url: Union[str, Path], path: Path, headers: Optional[dict] = None, num_attempts=1, overwrite=False
) -> pa.Schema:
    """
    Download a feather file from a URL to a local path.
    Returns the schema of feather file if successful.

    Parameters:
        url (str): URL to download feather file from.
        path (Path): Local path to save feather file to.
        headers (dict): Optional headers to include in request.
        num_attempts (int): Number of download attempts before raising an error.
        overwrite (bool): Whether to overwrite existing file.
    Returns:
        Feather schema.
    """
    assert num_attempts > 0, "Num attempts must be greater than 0"
    download_attempt = 0
    download_success = False
    schema = None
    while download_attempt < num_attempts and not download_success:
        download_attempt += 1
        if not path.exists() or overwrite:
            # Attempt download
            try:
                data = requests.get(str(url), headers=headers)
                readable = BytesIO(data.content)
                readable.seek(0)
                tb = pa.feather.read_table(readable, memory_map=False)  # type: ignore
                schema = tb.schema
                path.parent.mkdir(parents=True, exist_ok=True)
                pa.feather.write_feather(tb, path)  # type: ignore
                download_success = True
            except pa.ArrowInvalid:
                # failed try again
                path.unlink(missing_ok=True)
        else:
            # Load existing file
            try:
                schema = ipc.open_file(path).schema
                download_success = True
            except pa.ArrowInvalid:
                path.unlink(missing_ok=True)
    if not download_success or schema is None:
        raise ValueError(f"Failed to download feather file from {url} after {num_attempts} attempts.")
    return schema
