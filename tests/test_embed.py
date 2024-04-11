import pytest

from nomic import embed

parametrize = pytest.mark.parametrize

DEFAULT_MODEL = 'nomic-embed-text-v1'


def _test_local(text='x', n_tokens=1, dimensionality=None, **kwargs):
    result = embed.text([text], inference_mode='local', **kwargs)
    assert result['usage'] == {'prompt_tokens': n_tokens, 'total_tokens': n_tokens}
    assert result['model'] == DEFAULT_MODEL
    embedding, = result['embeddings']
    assert len(embedding) == dimensionality or 768  # n_embd


# TODO(jared): Empty string fails to embed due to tokenization issues.
#              Should be fixed by llama.cpp PR #6498
#@parametrize('text,n_tokens', [('The quick brown fox jumps over the lazy dog.', 10), ('', 0)])
@parametrize('text,n_tokens', [('The quick brown fox jumps over the lazy dog.', 10)])
def test_embed_local(text, n_tokens):
    _test_local(text, n_tokens)


def test_embed_empty_text():
    """embedding an empty string locally is not implemented"""
    with pytest.raises(NotImplementedError):
        embed.text([''], inference_mode='local')


def test_embed_empty_list():
    """embedding an empty list returns an empty result"""
    result = embed.text([], inference_mode='local')
    assert result['usage'] == {}
    assert result['model'] == DEFAULT_MODEL
    assert result['embeddings'] == []


@parametrize('task', ['search_query', 'search_document', 'classification', 'clustering'])
def test_embed_local_task(task):
    _test_local(task_type=task)


def text_embed_local_dimensionality():
    _test_local(model='nomic-embed-text-v1.5', dimensionality=64)
    _test_local(dimensionality=768)  # can be used with v1 if set to n_embd
    with pytest.warns(UserWarning):
        # this is allowed but not recommended
        _test_local(model='nomic-embed-text-v1.5', dimensionality=32)


def text_embed_non_matroyshka():
    """only nomic-embed-text-v1.5 can output a smaller dimensionality"""
    with pytest.raises(RuntimeError):
        embed.text(['x'], dimensionality=64)


def test_embed_bad_mode():
    with pytest.raises(ValueError):
        embed.text(['x'], inference_mode='foo')


@parametrize('mode', ['local', 'remote'])
def test_embed_bad_values(mode):
    """do not accept invalid argument values"""
    with pytest.raises(TypeError):
        # text argument must be a list, not str
        embed.text('x', inference_mode=mode)
    def check(**kwargs):
        return embed.text(['x'], inference_mode=mode, **kwargs)
    with pytest.raises(ValueError):
        check(model='foo')
    with pytest.raises(ValueError):
        check(dimensionality=0)
    with pytest.raises(ValueError):
        check(long_text_mode='foo')


def test_embed_remote_kwargs():
    """do not accept local kwargs in remote mode"""
    with pytest.raises(TypeError):
        embed.text('x', device='cpu')
    with pytest.raises(TypeError):
        embed.text('x', n_ctx=2048)
