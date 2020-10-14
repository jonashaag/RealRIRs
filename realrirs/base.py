import abc
import os
import pathlib
from typing import Generic, Iterable, List, NewType, Optional, Sequence, Tuple, TypeVar
import numpy as np

#: IR = Type of IR; numpy array of shape (chans, samples). Note that both `chans` and `samples` may be different for each IR of a dataset.
IR = NewType("IR", np.ndarray)
#: NAME_T = Type of IR name; eg., str or Tuple[str, int]
NAME_T = TypeVar("NAME_T")


def shape(things):
    """Get shape in a way that also understands Python lists."""
    try:
        return things.shape
    except AttributeError:
        shapes = list(map(shape, things))
        assert len(set(shapes)) == 1
        return (len(shapes), *shapes[0])


def check_nonmono(x):
    """Check that `x` is of shape `(chans, samples)`"""
    assert (
        len(shape(x)) == 2 and shape(x)[0] < 10
    ), f"Shape should be (channels, samples) but is {shape(x)}"


class IRDataset(Generic[NAME_T]):
    """Base class for all IR datasets.

    Each dataset contains many IRs.  An IR is a audio array of shape `(chans, samples)`,
    i.e. always a *non-mono* audio array.  Each IR has a "name" (of type str or any
    other type compatible with `__getitem__`).  IRs may be loaded from a dataset using
    `__getitem__`: ``ds[ir_name]``.

    Most datasets will be file-based, but that is not a requirement.
    """

    #: (required) Unique name for this dataset.
    name: str
    #: (optional) Where to find about more about this dataset.
    url: Optional[str] = None
    #: (optional) Direct dataset download URLs.
    download_urls: Sequence[str] = ()
    #: (optional) Copyright notice, list of authors, license name/URL, etc.
    license: Optional[str] = None

    @abc.abstractmethod
    def getall(self) -> Iterable[Tuple[NAME_T, int, IR]]:
        """All IRs in this dataset.

        The number of IRs returned must be the same as by list_irs(),
        and order must be identical.

        Returns:
            Iterator or sequence of (name: NAME_T, sr: int, ir: IR) pairs,
            where `sr` is the IR's sample rate.
        """

    @abc.abstractmethod
    def __getitem__(self, name: NAME_T) -> IR:
        """Get a single IR by name.

        Returns:
            The IR audio array
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of IRs in this dataset."""

    @abc.abstractmethod
    def list_irs(self) -> List[Tuple[NAME_T, int, int, int]]:
        """List of IRs in this dataset, with metadata.

        The number of IRs returned must be the same as by getall(),
        and order must be identical.

        Returns:
            List of (name, nchan, nsamples, sr) pairs, where `nchan` is the
            number of audio channels in the IR, `nsamples` is the number of
            samples (duration) of the IR, and `sr` is the IR's sample rate.
        """

    def __str__(self):
        return f"{self.__class__} ({self.name})"


class FileIRDataset(IRDataset[NAME_T]):
    """Base class for datasets whose IRs are read from files.

    Args:
        root (pathlib.Path or str): Root directory where the dataset resides.
    """

    file_patterns: Sequence[str]
    exclude_patterns: Sequence[str] = ()
    _files_list: List[pathlib.Path]

    def __init__(self, root: pathlib.Path):
        super().__init__()
        self.root = pathlib.Path(root)

    @abc.abstractmethod
    def _get_ir(self, name: NAME_T) -> IR:
        """Retrieve a single IR."""

    @abc.abstractmethod
    def _list_irs(self) -> List[Tuple[NAME_T, int, int, int]]:
        """See ``.list_irs``."""

    def __str__(self):
        return super().__str__() + f" root={self.root}"

    def list_files(self) -> List[pathlib.Path]:
        """List all files in the dataset."""
        self._populate_files_list()
        return self._files_list

    def _populate_files_list(self):
        if not hasattr(self, "_files_list"):
            self._files_list = self._list_files()

    def _list_files(self) -> List[pathlib.Path]:
        return [
            f
            for p in self.file_patterns
            for f in sorted(self.root.glob(p.replace("/", os.sep)))
            if not any(f.match(e) for e in self.exclude_patterns)
        ]

    def getall(self):
        self._populate_irs_list()
        for name, sr, ir in self._getall():
            check_nonmono(ir)
            yield name, sr, ir

    def __getitem__(self, name):
        self._populate_irs_list()
        ir = self._get_ir(name)
        check_nonmono(ir)
        return ir

    def __len__(self):
        return len(self.list_irs())

    def list_irs(self):
        self._populate_irs_list()
        return self._irs_list

    def _populate_irs_list(self):
        if not hasattr(self, "_irs_list"):
            self._irs_list = self._list_irs()

    def _getall(self):
        return ((name, sr, self[name]) for name, *_, sr in self.list_irs())


class CacheMixin:
    def __init__(self):
        self.__cache = {}

    def cached(self, key, func, *args, **kwargs):
        if key not in self.__cache:
            self.__cache[key] = func(*args, **kwargs)
        return self.__cache[key]
