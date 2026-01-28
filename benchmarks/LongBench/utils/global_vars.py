
_GLOBAL_ARGS = None

def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

def get_args():
    """Return arguments."""
    return _GLOBAL_ARGS