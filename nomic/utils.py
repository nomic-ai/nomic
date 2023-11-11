import base64
import gc
import sys
import pyarrow as pa
from uuid import UUID
from wonderwords import RandomWord


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
    random_words = RandomWord()
    return f"{random_words.word(include_parts_of_speech=['adjectives'])}-{random_words.word(include_parts_of_speech=['nouns'])}"


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
