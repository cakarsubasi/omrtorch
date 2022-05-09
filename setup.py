import setuptools

setuptools.setup(
    name="omrtorch-cakarsubasi",
    version="0.0.3",
    author="Cem Akarsubasi",
    author_email="cemakarsubasi@gmail.com",
    description="Music semantics extractor built via PyTorch.",
    url="https://github.com/cakarsubasi/omrtorch",
    project_urls={
        "Bug Tracker": "https://github.com/cakarsubasi/omrtorch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": ".",
    #            #"datasets": "omrmodules/datasets",
    #            },
    packages=setuptools.find_packages(),
    install_requires=[
        'opencv-contrib-python',
        'xmlschema',
        'omrdatasettools',
        'pycocotools',
        'music21',
    ],
    python_requires=">=3.7",
)