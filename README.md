## CosmicWebClassification 

### Description
Simple python code for classifying cosmic web structure in cosmological simulations [WIP] following [Hoffman et al 2012](https://academic.oup.com/mnras/article/425/3/2049/982860).

### Developers
Jordan Winstanley & Chris Power

### Installation
You can install as a package by doing the following:

```
git clone https://github.com/jwinstan/CosmicWebClassification.git
cd CosmicWebClassification
pip install -e .
```
This will load any dependencies needed by the scripts. 

You can call it from within python using 
```
from CosmicWebClassification.cosmic_web import CosmicWebClassifier
```

Class can be constructed via

```
web_classifier = CosmicWebClassifier(box_size, grid_size, method)
```
Currently available methods:
  - Nearest grid point "ngp"
  - Cloud in cell "cic"
  - Triangular shaped cloud "tsc"

Example: If you have a simulation box size of 1000 Mpc/h, and you want a grid size of 256 that uses cloud in cell interpolation (this assumes H0 is 67.5 km/s/Mpc).
```
web_classifier = CosmicWebClassifier(1000, 256, "cic")

```

Data can be added using (ensure that the positions are of the same units as the simulation box size).
```
web_classifier.add_batch(positions, velocities, masses)
```

Once all the data has been added the main routine can be executed by:
```
web = web_classifier.classify_structure()
```

A simple plot can be made with
```
web_classifier.plot("test.png")
```
<p align="center">
  <img src="test.png" alt="Cosmic Web Example" width="500"/>
</p>
