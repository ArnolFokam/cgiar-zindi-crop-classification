import contextlib
import os
from pathlib import Path
import timeit

def get_dir(*paths) -> str:
    """
    Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    
    Args:
        paths (List[str]): list of string that constitutes the path
    Returns:
        str: the created or existing path
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return Path(directory)

@contextlib.contextmanager
def time_activity(activity_name: str):
    """Times the execution of a piece of code
    Args:
        activity_name (str): string that describes the name of an activity. It can be arbitrary
    """
    print(f"[Timing] {activity_name} started.")
    start = timeit.default_timer()
    yield
    duration = timeit.default_timer() - start
    
    print(f"[Timing] {activity_name} finished (Took {duration:.2f}s).")