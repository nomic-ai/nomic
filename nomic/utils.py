from wonderwords import RandomWord
from uuid import UUID

def get_random_name():
    random_words = RandomWord()
    return f"{random_words.word(include_parts_of_speech=['adjectives'])}-{random_words.word(include_parts_of_speech=['nouns'])}"


def assert_valid_project_id(project_id):
    try:
        uuid_obj = UUID(project_id, version=4)
    except ValueError:
        raise ValueError(f"`{project_id}` is not a valid project id.")
