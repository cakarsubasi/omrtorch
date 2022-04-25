import setuptools

setuptools.setup(
    name="omrtorch-cakarsubasi",
    version="0.0.1",
    author="Cem Akarsubasi",
    author_email="cemakarsubasi@gmail.com",
    description="Test install.",
    url="https://github.com/cakarsubasi/omrtorch",
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