# RealRIRs

A collection of loaders for real (recorded) impulse response databases, because apparently people cannot to stick to a single way of storing audio data.

## Installation

Use ``pip install realrirs``. Depending on what databases you want to use, you will have to install additional dependencies. To install all dependencies, use ``pip install realrirs[full]``.

## Usage

```python
import pathlib
import realrirs.datasets

aachen_impulse_response_database = realrirs.datasets.AIRDataset(
    pathlib.Path("/path/to/AIR_1_4")  # Can also pass simple str
)

# List all IRs in database, tuples of (name, n_channels, n_samples, sample_rate)
aachen_impulse_response_database.list_irs()
# => [(PosixPath('/path/to/AIR_1_4/air_phone_stairway_hfrp_1.mat'), 1, 144000, 48000),
#     (PosixPath('/path/to/AIR_1_4/air_binaural_booth_1_1_1.mat'), 1, 32767, 48000)],
#     ...]

# Get single IR, ndarray of shape (n_channels, n_samples)
aachen_impulse_response_database[aachen_impulse_response_database.list_irs()[0][0]]
# => array([[ 1.32764243e-07, -2.18957279e-08,  1.28081465e-07, ...,
#             0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])

# Get all IRs, much faster than using [] (__getitem__) multiple times
aachen_impulse_response_database.getall()
# => <generator object FileIRDataset.getall at 0x11af88e40>

# Generator contains (name, sample_rate, ir) tuples.
next(aachen_impulse_response_database.getall())
# => (PosixPath('/path/to/AIR_1_4/air_binaural_aula_carolina_0_1_1_90_3.mat'), 48000,
#     array([[-2.73920884e-06, -3.49019781e-06, -1.70998298e-06, ..., -7.13979890e-11]]))
```

## Supported datasets

| Dataset | License | Number of IRs | Total IR duration | Total IR duration (all channels) |
|-|-|-|-|-|
| [360° Binaural Room Impulse Response (BRIR) Database for 6DOF spatial perception research](https://zenodo.org/record/2641166) | CC BY 4.0 | 1726 | 143.8 s | 292.0 s |
| [Aachen Impulse Response Database](https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/aachen-impulse-response-database/) | ? | 214 | 8.0 s | 8.0 s |
| [Audio Spatialisation for Headphones (ASH) Impulse Response Dataset](https://github.com/ShanonPearce/ASH-IR-Dataset) | CC BY-CC-SA 4.0 | 787 | 11.6 s | 23.1 s |
| [BUT Speech@FIT Reverb Database](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) | CC BY 4.0 | 2325 | 38.7 s | 38.7 s |
| [Concert Hall Impulse Responses – Pori, Finland](http://legacy.spa.aalto.fi/projects/poririrs/) | Custom, similar to CC BY-NC-SA | 90 | 5.7 s | 16.6 s |
| [DRR-scaled Individual Binaural Room Impulse Responses](https://zenodo.org/record/61072) | CC BY-NC-SA 4.0 | 4936 | 125.1 s | 250.3 s |
| [Database of Omnidirectional and B-Format Impulse Responses](http://isophonics.net/content/room-impulse-response-data-set) | ? | 2041 | 68.0 s | 68.0 s |
| [Dataset of In-The-Ear and Behind-The-Ear Binaural Room Impulse Responses](https://github.com/pyBinSim/HeadRelatedDatabase) | CC BY-NC 4.0 | 192 | 2.1 s | 4.3 s |
| [Dataset of measured binaural room impulse responses for use in an position-dynamic auditory augmented reality application](https://zenodo.org/record/1321996) | CC BY-NC 4.0 | 3888 | 129.6 s | 259.2 s |
| [EchoThief Impulse Response Library](http://www.echothief.com/downloads/) | ? | 115 | 2.5 s | 4.9 s |
| [Greg Hopkins IR 1 – Digital, Analog, Real Spaces](https://www.dropbox.com/sh/vjf5bsi28hcrkli/AAAmln01N4awOuclCi5q0DOia/Greg%20Hopkins%20IR%201%20-%20Digital%2C%20Analog%2C%20Real%20Spaces) | ? | 22 | 1.1 s | 2.2 s |
| [GTU-RIR Gebze Technical University Room Impulse Reponse Dataset](https://github.com/mehmetpekmezci/gtu-rir) | GPL | 15000 | 500 s | 500 s |
| [Impulse Response Database for HybridReverb2](https://github.com/jpcima/HybridReverb2-impulse-response-database) | CC BY-SA 4.0 | 472 | 16.5 s | 16.5 s |
| [Impulse Responses from the Bell Labs Varechoic Chamber](?) | ? | 12 | 0.2 s | 0.2 s |
| [METU SPARG Eigenmike em32 Acoustic Impulse Response Dataset v0.1.0](https://zenodo.org/record/2635758) | CC BY 4.0 | 8052 | 268.4 s | 268.4 s |
| [Multi-Channel Impulse Response Database](https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/multi-channel-impulse-response-database/) | ? | 1872 | 312.0 s | 312.0 s |
| [Multichannel Acoustic Reverberation Database at York](https://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/) | ? | 72 | 1.6 s | 1.6 s |
| [Open Acoustic Impulse Response (Open AIR) Library](https://openairlib.net/) | ? | 504 | 55.6 s | 181.9 s |
| [R-Prox RIR samples Darmstadt June 2017](https://zenodo.org/record/1209820) | CC BY 4.0 | 2313 | 84.7 s | 84.7 s |
| [REVERB challenge RealData](http://reverb2014.dereverberation.com/) | ? | 36 | 0.6 s | 4.8 s |
| [RIR samples Darmstadt and Helsinki, Summer-Autumn 2018](https://zenodo.org/record/1434786) | CC BY 4.0 | 1788 | 25.3 s | 25.3 s |
| [RWCP Sound Scene Database in Real Acoustical Environments](https://www.openslr.org/13/) | ? | 6758 | 64.6 s | 64.6 s |
| [Single- and Multichannel Audio Recordings Database (SMARD)](https://www.smard.es.aau.dk/) | ? | 1008 | 198.3 s | 198.3 s |
| [Statistics of natural reverberation enable perceptual separation of sound and space](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) | ? | 270 | 2.9 s | 2.9 s |
| [Surrey Binaural Room Impulse Response Measurements](https://github.com/IoSR-Surrey/RealRoomBRIRs) | MIT | 370 | 4.2 s | 4.2 s |
| [The IoSR listening room multichannel BRIR dataset](https://github.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs) | CC BY 4.0 | 3456 | 78.6 s | 157.3 s |
| [Voxengo Free Reverb Impulse Responses](https://www.voxengo.com/impulses/) | Custom, similar to CC BY-SA | 38 | 1.4 s | 2.7 s |
