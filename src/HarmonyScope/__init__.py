import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s"
)

def set_verbosity(verbose: int):
    """verbose=0→INFO、1→DEBUG"""
    if verbose >= 1:
        logging.getLogger("HarmonyScope").setLevel(logging.DEBUG)