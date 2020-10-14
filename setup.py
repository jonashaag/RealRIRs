from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="realrirs",
    version="0.1.0",
    author="Jonas Haag",
    author_email="jonas@lophus.org",
    url="https://github.com/jonashaag/RealRIRs",
    description="Python loaders for many Real Room Impulse Response databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="ISC",
    python_requires=">=3.4",
    extras_require={
        "full": ["librosa", "pysofaconventions", "scipy", "soundfile"],
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ISC License",
        "Operating System :: OS Independent",
    ],
)
