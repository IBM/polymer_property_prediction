import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polymer_property_prediction",
    version="1.0.10",
    author="flowdisc@br.ibm.com",
    author_email="flowdisc@br.ibm.com",
    description="Library to calculate the physical properties of molecules \
based on their SMILES representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/polymer_property_prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=['numpy',
                      'rdkit-pypi',
                      'pandas'],
    license='BSD 3-Clause License'
)
