import os
import pathlib
import textwrap

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
]

meta = {}

for ds in datasets:
    irs_list = ds.list_irs()
    meta[ds.name] = {
        "n_irs": len(irs_list),
        "total_duration": sum(
            n_samples / sr for _, n_channels, n_samples, sr in irs_list
        ),
        "total_duration_channels": sum(
            n_samples / sr * n_channels for _, n_channels, n_samples, sr in irs_list
        ),
        "license": ds.license,
        "url": ds.url,
    }

for ds_name, ds_meta in sorted(meta.items()):
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
