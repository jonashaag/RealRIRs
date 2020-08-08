import pathlib
import realrirs.datasets

R = pathlib.Path("/your/root/")

datasets = [
    realrirs.datasets.AIRDataset(R.joinpath("AIR_1_4")),
    realrirs.datasets.WavDataset(R.joinpath("DRR scaled BRIRs")),
    realrirs.datasets.ASHIRDataset(R.joinpath("ASH-IR-Dataset")),
    realrirs.datasets.HopkinsDataset(R.joinpath("Hopkins IR Library")),
    realrirs.datasets.FlacDataset(R.joinpath("HybridReverb2-impulse-response-database")),
    realrirs.datasets.IOSRRealRoomsDataset(R.joinpath("IOSR RealRoomBRIRs")),
    realrirs.datasets.WavDataset(R.joinpath("isophonics")),
    realrirs.datasets.KEMARDataset(R.joinpath("KEMAR")),
    realrirs.datasets.WavDataset(R.joinpath("poririrs")),
    realrirs.datasets.Reverb2014Dataset(R.joinpath("reverb2014")),
    realrirs.datasets.DarmstadtDataset(R.joinpath("RIR_samples_2018_summer-autumn")),
    realrirs.datasets.DarmstadtDataset(R.joinpath("RIRsc_Darmstadt_June")),
    realrirs.datasets.WavDataset(R.joinpath("spargair")),
    realrirs.datasets.WavDataset(R.joinpath("voxengo")),
    realrirs.datasets.WavDataset(R.joinpath("MARDY")),
    realrirs.datasets.BellVarechoicDataset(R.joinpath("varechoic")),
    realrirs.datasets.TUIInEarBehindEarDataset(R.joinpath("TUI_InEar_BehindEar_BRIR_dataset")),
    realrirs.datasets.RWCPDataset(R.joinpath("RWCP")),
    realrirs.datasets.BUTDataset(R.joinpath("BUT_ReverbDB_rel_19_06_RIR-Only")),
    realrirs.datasets.OpenAIRDataset(R.joinpath("openair")),
    realrirs.datasets.MIRDDataset(R.joinpath("MIRD")),
    realrirs.datasets.IOSRListeningRoomsDataset(R.joinpath("IoSR_ListeningRoom_BRIRs")),
    realrirs.datasets.FOAIRDataset(R.joinpath("360-BRIR-FOAIR-database")),
]
