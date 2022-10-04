from wonderwords import RandomWord

def get_random_name():
    random_words = RandomWord()
    return f"{random_words.word(include_parts_of_speech=['adjectives'])}-{random_words.word(include_parts_of_speech=['nouns'])}"