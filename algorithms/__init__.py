"""__init__.py for algorithms"""

from os.path import dirname, basename, isfile
import glob

# LOADS __all__ with all algorithm names
MODULES = glob.glob(dirname(__file__)+"/*.py")
__all__ = [basename(f)[:-3] for f in MODULES if isfile(f) and not f.endswith('__init__.py')]
