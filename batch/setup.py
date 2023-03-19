
import setuptools

REQUIRED_PACKAGES = [
  "apache-beam[gcp]",
  "tensorflow",
  "gensim",
  "fsspec",
  "gcsfs",
  "numpy",
  "keras",
  "nltk"
]

setuptools.setup(
    name="twitchstreaming",
    version="0.0.1-roger",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description="Troll detection by el Roger",
)
