import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import functools as ft

import mendeleev

from pymatgen.core import Molecule
from pymatgen.io.xyz import XYZ
from pymatgen.analysis.local_env import JmolNN

from tqdm import tqdm

xyz = XYZ.from_str("tmQM_X_short.xyz")
y = pd.read_csv("tmQM_y_short.csv", sep=";")


@ft.cache
def _element_featurizer(element):
    ele = mendeleev.element(element)
    #################### Modify this to add new attributes from Mendeleev
    attrs = [
        "atomic_number",
        "atomic_radius_rahm",
        "atomic_radius",
        "atomic_volume",
        "covalent_radius_cordero",
        "electron_affinity",
        "electronegativity_allen",
        "electronegativity_mulliken",
        "electrophilicity",
        "hardness",
        "mendeleev_number",
        "nvalence",
        "oxistates",
        "proton_affinity",
        "softness",
        "vdw_radius",
    ]
    ####################

    features = {attr: getattr(ele, attr) for attr in attrs}
    if "oxistates" in features:
        features["oxistate"] = features.pop("oxistates")[0]

    for method_name in (
        "nvalence",
        "softness",
        "electrophilicity",
        "hardness",
        "electronegativity_allen",
        "electronegativity_mulliken",
    ):
        if method_name in features:
            features[method_name] = features[method_name]()

    for ion_energy in [1, 2, 3]:
        features[f"ion_energy_{ion_energy}"] = ele.ionenergies.get(ion_energy, 0)

    return features


def element_featurizer(element, prefix=""):
    features = _element_featurizer(element)
    if prefix != "":
        features = {f"{prefix}_{k}": v for k, v in features.items()}

    return features


nn = JmolNN()
feat_list = []
for mol in tqdm(xyz.all_molecules, desc="Processing features..."):
    metal_i = max(range(len(mol.species)), key=lambda i: mol.species[i].Z)
    metal_site = mol.sites[metal_i]
    metal_species = mol.species[metal_i]
    nns = nn.get_nn(mol.get_boxed_structure(100, 100, 100), metal_i)
    dists = [np.linalg.norm(neighbor.coords - metal_site.coords) for neighbor in nns]
    nearest = nns[np.argmax(dists)]

    comp_data = [
        pd.Series(element_featurizer(el.symbol, "comp_mean")) * v
        for el, v in mol.composition.items()
    ]
    comp_data = sum(comp_data) / mol.composition.num_atoms

    # dist_mat = mol.distance_matrix
    descriptors = {}

    ###################### Modify this to add your own descriptors
    descriptors["num_neighbors"] = len(nns)
    descriptors["nearest_neighbor_dist"] = min(dists)
    descriptors["average_dist"] = np.mean(dists)
    descriptors.update(element_featurizer(metal_species.symbol, "metal"))
    descriptors.update(
        element_featurizer(nearest.species.elements[0].symbol, "closest_neighbor")
    )
    ######################
    feat_list.append(descriptors)

feats = pd.DataFrame(feat_list)
feats.to_csv("custom_features.csv")

y.to_csv("targets.csv")
