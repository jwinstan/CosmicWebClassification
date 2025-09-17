from setuptools import setup, find_packages

setup(
    name="CosmicWebClassification",
    version="0.1.0",
    packages=find_packages(),
    license="GPL-3.0",
    description="Classify cosmic web structure in cosmological simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Chris Power (jwin0686@uni.sydney.edu.au)",
    install_requires=["h5py",
                      "numpy",
		      "numba",
		      "scipy",
		      "matplotlib",
		      "plotly",
                     ],  

)
