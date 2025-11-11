from .packed_matcher import PackedMatcher

_MATCHER = None

def matcher() -> PackedMatcher:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = PackedMatcher()
    return _MATCHER

def rebuild_cache():
    global _MATCHER
    _MATCHER = PackedMatcher()

def rebuild_cache_async():
    # synchronous but fast
    rebuild_cache()
