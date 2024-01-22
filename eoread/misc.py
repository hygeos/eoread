from .utils.fileutils import filegen, LockFile
from .utils.fileutils import safe_move, PersistentList
import warnings

warnings.warn('Module `misc` is deprecated, please use module `fileutils`',
              DeprecationWarning)