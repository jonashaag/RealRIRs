import pathlib
from typing import Tuple

import numpy as np

from .base import CacheMixin, FileIRDataset


class DelayedImportError:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        raise ImportError(f"Please install {self.name!r} to perform this operation.")


try:
    import scipy.io as scipy_io
except ImportError:
    scipy_io = DelayedImportError("scipy")
try:
    import soundfile
except ImportError:
    soundfile = DelayedImportError("soundfile")
try:
    import pysofaconventions as sofa_conv
except ImportError:
    sofa_conv = DelayedImportError("pysofaconventions")
try:
    import librosa
except ImportError:
    librosa = DelayedImportError("librosa")


def _soundfile_info(f: pathlib.Path) -> Tuple[int, int, int]:
    with soundfile.SoundFile(str(f)) as fobj:
        return fobj.channels, len(fobj), fobj.samplerate


def _audioread_info(f: pathlib.Path) -> Tuple[int, int, int]:
    with librosa.core.audio.audioread.audio_open(str(f)) as fobj:
        return fobj.channels, fobj.duration * fobj.samplerate, fobj.samplerate


class FOAIRDataset(FileIRDataset[Tuple[str, int]], CacheMixin):
    name = "360° Binaural Room Impulse Response (BRIR) Database for 6DOF spatial perception research"
    url = "https://zenodo.org/record/2641166"
    license = "CC BY 4.0"

    file_patterns = ["**/*.mat", "**/*.wav"]

    def _list_irs(self):
        files = self.list_files()
        mat_irs = [
            ((f, i), 2, n_samples, 48000)
            for f, _, (n_samples, n_irs), _ in [
                (f, *scipy_io.whosmat(f)[0]) for f in files if f.match("*.mat")
            ]
            for i in range(n_irs)
        ]
        wav_irs = [((f, -1), *_soundfile_info(f)) for f in files if f.match("*.wav")]
        return mat_irs + wav_irs

    def _get_ir(self, name):
        fpath, idx = name
        if idx == -1:
            with soundfile.SoundFile(fpath) as fobj:
                return fobj.read().T
        else:
            mat = self.cached(("loadmat", fpath), scipy_io.loadmat, fpath)
            return mat["IR_L"][:, idx], mat["IR_R"][:, idx]


class AIRDataset(FileIRDataset[str]):
    name = "Aachen Impulse Response Database"
    url = "https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/aachen-impulse-response-database/"
    license = "?"

    file_patterns = ["**/*.mat"]

    def _list_irs(self):
        files = self.list_files()
        return [
            (
                f,
                *[
                    info
                    for name, info, _ in scipy_io.whosmat(f, struct_as_record=False)
                    if name == "h_air"
                ][0],
                48000,
            )
            for f in files
        ]

    def _get_ir(self, name):
        return scipy_io.loadmat(name, struct_as_record=False)["h_air"]


class SoundfileDataset(FileIRDataset[pathlib.Path]):
    """Base class for datasets that can be read by `soundfile`."""

    def _list_irs(self):
        files = self.list_files()
        return [(f, *_soundfile_info(f)) for f in files]

    def _get_ir(self, name):
        with soundfile.SoundFile(name) as fobj:
            data = fobj.read()
            if data.ndim == 1:
                return data.reshape((1, -1))
            else:
                return data.T


class LibrosaDataset(FileIRDataset[pathlib.Path]):
    """Base class for datasets that can be read by `librosa` (most audio files)."""

    def _list_irs(self):
        files = self.list_files()
        return [(f, *_audioread_info(f)) for f in files]

    def _get_ir(self, name):
        data = librosa.core.load(name, sr=None, mono=False)[0]
        if data.ndim == 1:
            return data.reshape((1, -1))
        else:
            return data


class WavDataset(SoundfileDataset):
    file_patterns = ["**/*.wav"]


class FlacDataset(LibrosaDataset):
    file_patterns = ["**/*.flac"]


class ASHIRDataset(WavDataset):
    name = "Audio Spatialisation for Headphones (ASH) Impulse Response Dataset"
    url = "https://github.com/ShanonPearce/ASH-IR-Dataset"
    license = "CC BY-CC-SA 4.0"

    file_patterns = ["BRIRs/**/*.wav"]


class HopkinsDataset(WavDataset):
    name = "Greg Hopkins IR 1 – Digital, Analog, Real Spaces"
    url = "https://www.dropbox.com/sh/vjf5bsi28hcrkli/AAAmln01N4awOuclCi5q0DOia/Greg%20Hopkins%20IR%201%20-%20Digital%2C%20Analog%2C%20Real%20Spaces"
    license = "?"

    file_patterns = ["Real Spaces/**/*.wav"]


class IOSRRealRoomsDataset(FileIRDataset[Tuple[pathlib.Path, int, int]], CacheMixin):
    name = "Surrey Binaural Room Impulse Response Measurements"
    url = "https://github.com/IoSR-Surrey/RealRoomBRIRs"
    license = "MIT"

    file_patterns = ["**/*_48k.sofa"]

    def _list_irs(self):
        files = self.list_files()
        return [
            ((f, measurement, receiver), 1, dims["N"].size, 48000)
            for f, dims in [
                (f, sofa_conv.SOFAFile(str(f), "r").getDimensionsAsDict())
                for f in files
            ]
            for measurement in range(dims["M"].size)
            for receiver in range(dims["R"].size)
        ]

    def _get_ir(self, name):
        fpath, measurement, receiver = name
        return self.cached(fpath, sofa_conv.SOFAFile(str(fpath), "r").getDataIR)[
            measurement, receiver
        ].reshape((1, -1))


class KEMARDataset(FileIRDataset[Tuple[pathlib.Path, str]]):
    name = "Dataset of measured binaural room impulse responses for use in an position-dynamic auditory augmented reality application"
    url = "https://zenodo.org/record/1321996"
    license = "CC BY-NC 4.0"

    file_patterns = ["**/*.mat"]

    def _list_irs(self):
        files = self.list_files()
        surround_types = ["L", "LS", "R", "RS", "C", "S"]
        return [((f, t), 2, 96000, 48000) for f in files for t in surround_types]

    def _get_ir(self, name):
        fpath, surround_type = name
        irs = scipy_io.loadmat(fpath, struct_as_record=False)["brirData"][0][
            0
        ].impulseResponse
        (ir,) = [ir for t, ir in irs if t[0] == surround_type]
        return ir.T

    def _getall(self):
        for f in self.list_files():
            irs = scipy_io.loadmat(f, struct_as_record=False)["brirData"][0][
                0
            ].impulseResponse
            for t, ir in irs:
                yield (f, t[0]), ir.T


class MIRDDataset(FileIRDataset[Tuple[pathlib.Path, int]]):
    name = "Multi-Channel Impulse Response Database"
    url = "https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/multi-channel-impulse-response-database/"
    license = "?"

    file_patterns = ["**/*.mat"]

    def _list_irs(self):
        files = self.list_files()
        return [((f, i), 1, 480000, 48000) for f in files for i in range(8)]

    def _get_ir(self, name):
        fpath, i = name
        return scipy_io.loadmat(fpath, struct_as_record=False)["impulse_response"][
            :, i
        ].reshape((1, -1))

    def _getall(self):
        for f in self.list_files():
            irs = scipy_io.loadmat(f, struct_as_record=False)["impulse_response"]
            for idx, ir in enumerate(irs.T):
                yield (f, idx), ir.reshape((1, -1))


class Reverb2014Dataset(WavDataset):
    name = "REVERB challenge RealData"
    url = "http://reverb2014.dereverberation.com/"
    download_urls = [
        "http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz"
    ]
    license = "?"

    file_patterns = ["**/RIR_*.wav"]


class TUIInEarBehindEarDataset(FileIRDataset[Tuple[pathlib.Path, str, int]]):
    name = "Dataset of In-The-Ear and Behind-The-Ear Binaural Room Impulse Responses"
    url = "https://github.com/pyBinSim/HeadRelatedDatabase"
    license = "CC BY-NC 4.0"

    file_patterns = ["lab_brirs.mat", "reha_brirs.mat", "tvstudio_brirs.mat"]

    def _list_irs(self):
        lab, reha, tvstudio = self.list_files()
        cfgs = [(t, i) for t in ["inear", "btear"] for i in range(32)]
        return (
            [((lab, t, i), 2, 22050, 44100) for t, i in cfgs]
            + [((reha, t, i), 2, 44100, 44100) for t, i in cfgs]
            + [((tvstudio, t, i), 2, 22050, 44100) for t, i in cfgs]
        )

    def _get_ir(self, name):
        fpath, t, i = name
        mat = scipy_io.loadmat(fpath, struct_as_record=False)
        mat = getattr(mat[list(mat.keys())[-1]][0][0], t)[0][0]
        return mat.left[i], mat.right[i]

    def _getall(self):
        for f in self.list_files():
            mat = scipy_io.loadmat(f, struct_as_record=False)
            mat = mat[list(mat.keys())[-1]][0][0]
            for t in ["inear", "btear"]:
                data = getattr(mat, t)[0][0]
                for idx, (l, r) in enumerate(zip(data.left, data.right)):
                    yield (f, t, idx), (l, r)


class BellVarechoicDataset(FileIRDataset[Tuple[pathlib.Path, int]]):
    name = "Impulse Responses from the Bell Labs Varechoic Chamber"
    url = "?"
    license = "?"

    def _list_files(self):
        return [
            self.root.joinpath("IR_00.mat"),
            self.root.joinpath("IR_43.mat"),
            self.root.joinpath("IR_100.mat"),
        ]

    def _list_irs(self):
        files = self.list_files()
        return [((f, i), 1, 8192, 10000) for i in range(4) for f in files]

    def _get_ir(self, name):
        fpath, i = name
        return (
            list(scipy_io.loadmat(fpath).values())[0][:, i]
            .astype("float64")
            .reshape((1, -1))
        )


class IOSRListeningRoomsDataset(FileIRDataset[Tuple[pathlib.Path, int]], CacheMixin):
    name = "The IoSR listening room multichannel BRIR dataset"
    url = "https://github.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs"
    license = "CC BY 4.0"

    file_patterns = ["IoSR_ListeningRoom_BRIRs.sofa"]

    def _list_irs(self):
        files = self.list_files()
        return [
            ((f, measurement), dims["R"].size, dims["N"].size, 48000)
            for f, dims in [
                (f, sofa_conv.SOFAFile(str(f), "r").getDimensionsAsDict())
                for f in files
            ]
            for measurement in range(dims["M"].size)
        ]

    def _get_ir(self, name):
        fpath, measurement = name
        return self.cached(fpath, sofa_conv.SOFAFile(str(fpath), "r").getDataIR)[
            measurement
        ]


class BinaryArrayDataset(FileIRDataset[pathlib.Path]):
    """Base class for datasets that are stored as binary audio arrays and can
    be read using ``np.fromfile`` or ``np.memmap``.

    Args:
        use_memmap (bool, default True): Whether to use ``np.memmap`` to read the files.
    """

    #: Sample rate of the array
    sample_rate: int
    #: Data type of the array (str or dtype object)
    dtype = "float32"

    def __init__(self, *args, **kwargs):
        self.use_memmap = kwargs.pop("use_memmap", True)
        super().__init__(*args, **kwargs)

    def _list_irs(self):
        files = self.list_files()
        return [
            (f, 1, f.stat().st_size / np.dtype(self.dtype).itemsize, self.sample_rate)
            for f in files
        ]

    def _get_ir(self, name):
        if self.use_memmap:
            return np.memmap(name, self.dtype, "r").reshape((1, -1))
        else:
            return np.fromfile(name, self.dtype).reshape((1, -1))


class RWCPDataset(BinaryArrayDataset):
    name = "RWCP Sound Scene Database in Real Acoustical Environments"
    url = "https://www.openslr.org/13/"
    license = "?"

    sample_rate = 48000
    file_patterns = ["near/data/rsp*/*", "micarray/**/imp*.*"]

    def _get_ir(self, name):
        data = super()._get_ir(name)
        # Normalize very large values
        return data / np.abs(data).max()


class BUTDataset(WavDataset):
    name = "BUT Speech@FIT Reverb Database"
    url = "https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database"
    license = "CC BY 4.0"

    file_patterns = ["**/IR_*.wav"]


class OpenAIRDataset(WavDataset):
    name = "Open Acoustic Impulse Response (Open AIR) Library"
    url = "https://openairlib.net/"
    license = "?"

    exclude_patterns = ["examples/*"]


class Darmstadt2017SamplesDataset(WavDataset):
    name = "R-Prox RIR samples Darmstadt June 2017"
    url = "https://zenodo.org/record/1209820"
    license = "CC BY 4.0"

    file_patterns = ["**/*rir.wav"]


class Darmstadt2018SamplesDataset(WavDataset):
    name = "RIR samples Darmstadt and Helsinki, Summer-Autumn 2018"
    url = "https://zenodo.org/record/1434786"
    license = "CC BY 4.0"

    file_patterns = ["**/*rir.wav"]


class DRRDataset(WavDataset):
    name = "DRR-scaled Individual Binaural Room Impulse Responses"
    url = "https://zenodo.org/record/61072"
    license = "CC BY-NC-SA 4.0"


class IsophonicsDataset(WavDataset):
    name = "Database of Omnidirectional and B-Format Impulse Responses"
    url = "http://isophonics.net/content/room-impulse-response-data-set"
    license = "?"


class PoriIRsDataset(WavDataset):
    name = "Concert Hall Impulse Responses – Pori, Finland"
    url = "http://legacy.spa.aalto.fi/projects/poririrs/"
    license = "Custom, similar to CC BY-NC-SA"


class SPARGAIRDataset(WavDataset):
    name = "METU SPARG Eigenmike em32 Acoustic Impulse Response Dataset v0.1.0"
    url = "https://zenodo.org/record/2635758"
    license = "CC BY 4.0"


class VoxengoDataset(WavDataset):
    name = "Voxengo Free Reverb Impulse Responses"
    url = "https://www.voxengo.com/impulses/"
    license = "Custom, similar to CC BY-SA"


class MARDYDataset(WavDataset):
    name = "Multichannel Acoustic Reverberation Database at York"
    url = "https://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/"
    license = "?"


class HybridReverb2Dataset(FlacDataset):
    name = "Impulse Response Database for HybridReverb2"
    url = "https://github.com/jpcima/HybridReverb2-impulse-response-database"
    license = "CC BY-SA 4.0"


class MITDataset(WavDataset):
    name = "Statistics of natural reverberation enable perceptual separation of sound and space"
    url = "https://mcdermottlab.mit.edu/Reverb/IR_Survey.html"
    license = "?"
