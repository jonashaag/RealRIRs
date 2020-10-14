import os
import pathlib
import textwrap

import numpy as np

import realrirs.datasets

R = pathlib.Path(os.environ["REALRIRS_ROOT"])

datasets = [
    realrirs.datasets.AIRDataset(R.joinpath("AIR_1_4")),
    realrirs.datasets.DRRDataset(R.joinpath("DRR scaled BRIRs")),
    realrirs.datasets.ASHIRDataset(R.joinpath("ASH-IR-Dataset")),
    realrirs.datasets.HopkinsDataset(R.joinpath("Hopkins IR Library")),
    realrirs.datasets.HybridReverb2Dataset(
        R.joinpath("HybridReverb2-impulse-response-database")
    ),
    realrirs.datasets.IOSRRealRoomsDataset(R.joinpath("IOSR RealRoomBRIRs")),
    realrirs.datasets.IsophonicsDataset(R.joinpath("isophonics")),
    realrirs.datasets.KEMARDataset(R.joinpath("KEMAR")),
    realrirs.datasets.PoriIRsDataset(R.joinpath("poririrs")),
    realrirs.datasets.Reverb2014Dataset(R.joinpath("reverb2014")),
    realrirs.datasets.Darmstadt2018SamplesDataset(
        R.joinpath("RIR_samples_2018_summer-autumn")
    ),
    realrirs.datasets.Darmstadt2017SamplesDataset(R.joinpath("RIRsc_Darmstadt_June")),
    realrirs.datasets.SPARGAIRDataset(R.joinpath("spargair")),
    realrirs.datasets.VoxengoDataset(R.joinpath("voxengo")),
    realrirs.datasets.MARDYDataset(R.joinpath("MARDY")),
    realrirs.datasets.BellVarechoicDataset(R.joinpath("varechoic")),
    realrirs.datasets.TUIInEarBehindEarDataset(
        R.joinpath("TUI_InEar_BehindEar_BRIR_dataset")
    ),
    realrirs.datasets.RWCPDataset(R.joinpath("RWCP")),
    realrirs.datasets.BUTDataset(R.joinpath("BUT_ReverbDB_rel_19_06_RIR-Only")),
    realrirs.datasets.OpenAIRDataset(R.joinpath("openair")),
    realrirs.datasets.MIRDDataset(R.joinpath("MIRD")),
    realrirs.datasets.IOSRListeningRoomsDataset(R.joinpath("IoSR_ListeningRoom_BRIRs")),
    realrirs.datasets.FOAIRDataset(R.joinpath("360-BRIR-FOAIR-database")),
    realrirs.datasets.MITDataset(R.joinpath("MIT")),
    realrirs.datasets.EchoThiefDataset(R.joinpath("EchoThiefImpulseResponseLibrary")),
    realrirs.datasets.SMARDDataset(R.joinpath("SMARD")),
]


def process_ds(ds):
    print("Processing", ds)
    trimmed_ir_shapes = [
        (ir.shape[0], len(np.trim_zeros(ir[0])) / sr) for _, sr, ir in ds.getall()
    ]
    return ds.name, {
        "n_irs": len(trimmed_ir_shapes),
        "total_duration": sum(1 * trimmed_len for _, trimmed_len in trimmed_ir_shapes),
        "total_duration_channels": sum(
            n_channels * trimmed_len for n_channels, trimmed_len in trimmed_ir_shapes
        ),
        "license": ds.license,
        "url": ds.url,
    }


for ds_name, ds_meta in sorted(map(process_ds, datasets)):
    print(
        " | ".join(
            [
                "",
                f'[{ds_name}]({ds_meta["url"]})',
                ds_meta["license"] or "",
                str(ds_meta["n_irs"]),
                f'{ds_meta["total_duration"]/60:.1f} s',
                f'{ds_meta["total_duration_channels"]/60:.1f} s',
                "",
            ]
        ).strip()
    )
