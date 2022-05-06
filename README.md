# Polymer Property Prediction

A Python Library that calculates the physical properties of molecules based on their SMILES representations.

## Dependencies

* [NumPy](https://numpy.org) is the fundamental package for scientific computing with Python.
* [RDKit](https://www.rdkit.org/) is a collection of cheminformatics and machine-learning software written in C++ and Python.
* [Pandas](https://pandas.pydata.org/) is an open source data analysis and manipulation tool, built on top of the Python programming language.

## Developer tips

These tips are not mandatory, but they are a sure way of helping you develop the code while maintaining consistency with the current style, structure and formatting choices.

### Coding style guide

We recommend these tools to ensure code style compatibility.

* [autopep8](https://pypi.org/project/autopep8/) automatically formats Python code to conform to the PEP8 style guide.
* [Flake8](https://flake8.pycqa.org) is your tool for style guide enforcement.

## Installation

### Option 1: Using `setup.py`

Clone `polymer_property_prediction` repository if you haven't done it yet.

Go to `polymer_property_prediction`'s root folder, there you will find `setup.py` file, and run the command below:

```Shell
python setup.py install
```

### Option 2: Using pip/pipenv to install from Pypi.org

If you intend to use `pipenv`, please add the following to your `Pipfile`:

```Pipfile
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
polymer_property_prediction = "*"
```

If you intend to use `pip`, please run the command below:

```Shell
pip install polymer_property_prediction
```

### Option 3: Using pip to install directly from the GitHub repo

You can run

```Shell
pip install git+https://github.com/IBM/polymer_property_prediction.git
```

and then you will be prompted to enter your GitHub username and password/access token.

If you already have a SSH key configured, you can run

```Shell
pip install git+ssh://git@github.com/IBM/polymer_property_prediction.git
```

### Option 4: Using pip/pipenv to install from Artifactory

Log into Artifactory and access your user profile. There you will find your API key and username. Then export your credentials as environment variables for later use in the installation process.

```Shell
export ARTIFACTORY_USERNAME=username@email.com
export ARTIFACTORY_API_KEY=your-api-key
export ARTIFACTORY_URL=your-artifactory-url
```

If you intend to use `pipenv`, please add the following to your `Pipfile`:

```Pipfile
[[source]]
url = "https://$ARTIFACTORY_USERNAME:$ARTIFACTORY_API_KEY@$ARTIFACTORY_URL"
verify_ssl = true
name = "artifactory"

[packages]
polymer_property_prediction = {version="*", index="artifactory"}
```

If you intend to use `pip`, please run the command below:

```Shell
pip install polymer_property_prediction --extra-index-url=https://$ARTIFACTORY_USERNAME:$ARTIFACTORY_API_KEY@$ARTIFACTORY_URL
```

## Usage example

This is a small example of how to use our package:

```Python
>>> from polymer_property_prediction import polymer_properties_from_smiles as ppf
>>> smiles_opsin = '[*:1]CC(=O)O[*:2]'
>>> ppf.ConvertOpsinToMolSmiles(smiles_opsin)
'CC(=O)O'
>>> ppf.HeadTailAtoms(smiles_opsin)
array([0, 3], dtype=int32)
>>> smiles_mol = ppf.ConvertOpsinToMolSmiles(smiles_opsin)
>>> mol = ppf.Chem.MolFromSmiles(smiles_mol)
```

You can also access our [tutorial](polymer_property_prediction_tutorial.ipynb).

## Python package deployment

### Deploying to Artifactory

We have an automated CI/CD pipeline running on TravisCI that takes every single `git push` event and executes the build/test/deploy instructions in the `.travis.yml`. If you are deploying `master` or `release` branches, a Python package will be generated and published to a private Pypi registry on Artifactory.

### Deploying to Pypi

We have an automated CI/CD pipeline running on TravisCI that takes every single `git push` event and executes the build/test/deploy instructions in the `.travis.yml`. If you are deploying `main` branch, a Python package will be generated and published to Pypi.org registry.
