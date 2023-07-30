from typing import Any, Callable

import asyncio
from syncer import sync
from asyncer import asyncify

def make_async(function: Callable):

    def wrapper(*args, **kwargs):
        res = function(*args, **kwargs)
        return res

    return asyncify(wrapper, cancellable=True)