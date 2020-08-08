import numpy as np
import pysofaconventions as sofa_conv
import scipy
import soundfile as sf

from .base import CacheMixin, IRDataset, sf_info


class FOAIRDataset(IRDataset, CacheMixin):
    name = "360° Binaural Room Impulse Response (BRIR) Database for 6DOF spatial perception research"
    url = "https://zenodo.org/record/2641166"
    license = "CC-BY-4.0"

    file_patterns = ["**/*.mat", "**/*.wav"]

    def _list_irs(self):
        files = self.list_files()
        mat_irs = [
            ((f, i), 2, n_samples, 48000)
            for f, _, (n_samples, n_irs), _ in [
                (f, *scipy.io.whosmat(f)[0]) for f in files if f.match("*.mat")
            ]
            for i in range(n_irs)
        ]
        wav_irs = [
            (f, *sf_info(f, ["channels", "frames", "samplerate"]))
            for f in files
            if f.match("*.wav")
        ]
        return mat_irs + wav_irs

    def _getir(self, name):
        if isinstance(name, tuple):
            fpath, idx = name
            mat = self.cached(("loadmat", fpath), scipy.io.loadmat, fpath)
            return mat["IR_L"][:, idx], mat["IR_R"][:, idx]
        else:
            with sf.SoundFile(name) as fobj:
                return fobj.read().T


class AIRDataset(IRDataset):
    file_patterns = ["**/*.mat"]

    def _list_irs(self):
        files = self.list_files()
        return [
            (
                f,
                *[
                    info
                    for name, info, _ in scipy.io.whosmat(f, struct_as_record=False)
                    if name == "h_air"
                ][0],
                48000,
            )
            for f in files
        ]

    def _getir(self, name):
        return scipy.io.loadmat(name, struct_as_record=False)["h_air"]


class SoundfileDataset(IRDataset):
    def _list_irs(self):
        files = self.list_files()
        return [(f, *sf_info(f, ["channels", "frames", "samplerate"])) for f in files]

    def _getir(self, name):
        with sf.SoundFile(name) as fobj:
            data = fobj.read()
            if len(data.shape) == 1:
                return data.reshape((1, -1))
            else:
                return data.T


class WavDataset(SoundfileDataset):
    file_patterns = ["**/*.wav"]


class FlacDataset(SoundfileDataset):
    file_patterns = ["**/*.flac"]


class ASHIRDataset(WavDataset):
    file_patterns = ["BRIRs/**/*.wav"]


class HopkinsDataset(WavDataset):
    file_patterns = ["Real Spaces/**/*.wav"]


class IOSRRealRoomsDataset(IRDataset, CacheMixin):
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

    def _getir(self, name):
        fpath, measurement, receiver = name
        return self.cached(fpath, sofa_conv.SOFAFile(str(fpath), "r").getDataIR)[
            measurement, receiver
        ].reshape((1, -1))


class KEMARDataset(IRDataset):
    file_patterns = ["**/*.mat"]

    def _list_irs(self):
        files = self.list_files()
        surround_types = ["L", "LS", "R", "RS", "C", "S"]
        return [((f, t), 2, 96000, 48000) for f in files for t in surround_types]

    def _getir(self, name):
        fpath, surround_type = name
        irs = scipy.io.loadmat(fpath, struct_as_record=False)["brirData"][0][
            0
        ].impulseResponse
        (ir,) = [ir for t, ir in irs if t[0] == surround_type]
        return ir.T

    def _getall(self):
        for f in self.list_files():
            irs = scipy.io.loadmat(f, struct_as_record=False)["brirData"][0][
                0
            ].impulseResponse
            for t, ir in irs:
                yield (f, t[0]), ir.T


class MIRDDataset(IRDataset):
    file_patterns = ["**/*.mat"]

    def _list_irs(self):
        files = self.list_files()
        return [((f, i), 1, 480000, 48000) for f in files for i in range(8)]

    def _getir(self, name):
        fpath, i = name
        return scipy.io.loadmat(fpath, struct_as_record=False)["impulse_response"][
            :, i
        ].reshape((1, -1))

    def _getall(self):
        for f in self.list_files():
            irs = scipy.io.loadmat(f, struct_as_record=False)["impulse_response"]
            for idx, ir in enumerate(irs.T):
                yield (f, idx), ir.reshape((1, -1))


class Reverb2014Dataset(WavDataset):
    file_patterns = ["**/RIR_*.wav"]


class TUIInEarBehindEarDataset(IRDataset):
    file_patterns = ["lab_brirs.mat", "reha_brirs.mat", "tvstudio_brirs.mat"]

    def _list_irs(self):
        lab, reha, tvstudio = self.list_files()
        cfgs = [(t, i) for t in ["inear", "btear"] for i in range(32)]
        return (
            [((lab, t, i), 2, 22050, 44100) for t, i in cfgs]
            + [((reha, t, i), 2, 44100, 44100) for t, i in cfgs]
            + [((tvstudio, t, i), 2, 22050, 44100) for t, i in cfgs]
        )

    def _getir(self, name):
        fpath, t, i = name
        mat = scipy.io.loadmat(fpath, struct_as_record=False)
        mat = getattr(mat[list(mat.keys())[-1]][0][0], t)[0][0]
        return mat.left[i], mat.right[i]

    def _getall(self):
        for f in self.list_files():
            mat = scipy.io.loadmat(f, struct_as_record=False)
            mat = mat[list(mat.keys())[-1]][0][0]
            for t in ["inear", "btear"]:
                data = getattr(mat, t)[0][0]
                for idx, (l, r) in enumerate(zip(data.left, data.right)):
                    yield (f, t, idx), (l, r)


class BellVarechoicDataset(IRDataset):
    def _list_files(self):
        return [
            self.root.joinpath("IR_00.mat"),
            self.root.joinpath("IR_43.mat"),
            self.root.joinpath("IR_100.mat"),
        ]

    def _list_irs(self):
        files = self.list_files()
        return [((f, i), 1, 8192, 10000) for i in range(4) for f in files]

    def _getir(self, name):
        fpath, i = name
        return (
            list(scipy.io.loadmat(fpath).values())[0][:, i]
            .astype("float64")
            .reshape((1, -1))
        )


class IOSRListeningRoomsDataset(IRDataset, CacheMixin):
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

    def _getir(self, name):
        fpath, measurement = name
        return self.cached(fpath, sofa_conv.SOFAFile(str(fpath), "r").getDataIR)[
            measurement
        ]


class BinaryArrayDataset(IRDataset):
    dtype = "float32"

    def __init__(self, *args, **kwargs):
        self.use_memmap = kwargs.pop("use_memmap", True)
        super().__init__(*args, **kwargs)

    def _list_irs(self):
        files = self.list_files()
        return [
            (f, 1, f.stat().st_size / np.dtype(self.dtype).name, self.sample_rate)
            for f in files
        ]

    def _getir(self, name):
        if self.use_memmap:
            return np.memmap(name, self.dtype, "r").reshape((1, -1))
        else:
            return np.fromfile(name, self.dtype).reshape((1, -1))


class RWCPDataset(BinaryArrayDataset):
    sample_rate = 48000
    file_patterns = ["near/data/rsp*/*", "micarray/**/imp*.*"]

    def _getir(self, name):
        data = super()._getir(name)
        # Normalize very large values
        return data / np.abs(data).max()


class BUTDataset(WavDataset):
    file_patterns = ["**/IR_*.wav"]


class OpenAIRDataset(WavDataset):
    exclude_patterns = ["examples/*"]


class DarmstadtDataset(WavDataset):
    file_patterns = ["**/*rir.wav"]


class DRRDataset(WavDataset):
    pass


class IsophonicsDataset(WavDataset):
    pass


class PoriIRsDataset(WavDataset):
    pass


class SPARGAIRDataset(WavDataset):
    pass


class VoxengoDataset(WavDataset):
    pass


class MARDYDataset(WavDataset):
    pass
