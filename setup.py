import setuptools

setuptools.setup(
    name="omrtorch-cakarsubasi",
    version="0.0.2",
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
    package_dir={"": ".",
                #"datasets": "omrmodules/datasets",
                },
    packages=['omrmodules'],
    python_requires=">=3.7",
)