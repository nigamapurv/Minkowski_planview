from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import io
import pstats

cache = {}

PROFILE_SWITCH = True

class Profiler:
    def __init__(self):
        self.profiler = cProfile.Profile()

    def profile(self, func, *args, **kwargs):
        self.profiler.enable()

        result = func(*args, **kwargs)

        self.profiler.disable()
        with io.StringIO() as string_stream:
            profiler_stats = pstats.Stats(self.profiler, stream = string_stream).sort_stats("cumulative")
            profiler_stats.print_stats()
            print(string_stream.getvalue())

        return result

class PassThroughProfiler:
    def __init__(self):
        pass

    def profile(self, func, *args, **kwargs):
        return func(*args, **kwargs)

def get_profiler():
    if "profiler" not in cache:
        if PROFILE_SWITCH:
            cache["profiler"] = Profiler()
        else:
            cache["profiler"] = PassThroughProfiler()

    return cache["profiler"]

def profile(func, *args, **kwargs):
    profiler = get_profiler()

    return profiler.profile(func, *args, **kwargs)

def profileable(func):
    def _profile(*args, **kwargs):
        return profile(func, *args, **kwargs)

    return _profile