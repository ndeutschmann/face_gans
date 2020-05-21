def cycle(iterable):
    """Create an infinite iterable from an iterable that cycles back to the beginning when reaching the end"""
    while True:
        for x in iterable:
            yield x