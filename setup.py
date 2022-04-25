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
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",
)