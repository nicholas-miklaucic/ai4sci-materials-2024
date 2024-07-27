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

xyz = XYZ.from_file("tmQM_X_short.xyz")
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
        "block",
        "covalent_radius_bragg",
        "covalent_radius_cordero",
        "electron_affinity",
        "electronegativity_allen",
        "electronegativity_martynov_batsanov",
        "electronegativity_mulliken",
        "electronegativity_sanderson",
        "electrophilicity",
        "group",
        "hardness",
        "mendeleev_number",
        "metallic_radius",
        "metallic_radius_c12",
        "nvalence",
        "oxistates",
        "proton_affinity",
        "period",
        "series",
        "softness",
        "symbol",
        "vdw_radius",
        "vdw_radius_bondi",
    ]
    ####################

    features = {attr: getattr(ele, attr) for attr in attrs}
    if "oxistates" in features:
        features["oxistate"] = features.pop("oxistates")[0]

    if "group" in features:
        features["group_symbol"] = features["group"].symbol
        features["group"] = features["group"].group_id

    for method_name, feature in features.items():
        if hasattr(feature, "__call__"):
            features[method_name] = feature()

    for ion_energy in [1, 2, 3]:
        features[f"ion_energy_{ion_energy}"] = ele.ionenergies.get(ion_energy, np.nan)

    return features


def element_featurizer(element, prefix=""):
    features = _element_featurizer(element)
    if prefix != "":
        features = {f"{prefix}_{k}": v for k, v in features.items()}

    return features


nn = JmolNN()
feat_list = []
for mol in tqdm(xyz.all_molecules, desc="Processing features..."):
    metal_inds = [i for i, s in enumerate(mol.species) if s.is_metal]
    assert len(metal_inds) == 1
    metal_i = metal_inds[0]
    metal_site = mol.sites[metal_i]
    metal_species = mol.species[metal_i]
    nns = nn.get_nn(mol.get_boxed_structure(100, 100, 100), metal_i)
    dists = [np.linalg.norm(neighbor.coords - metal_site.coords) for neighbor in nns]
    nearest = nns[np.argmax(dists)]

    comp_data = []
    for k, v in mol.composition.get_el_amt_dict().items():
        data = element_featurizer(k, "comp")
        for _i in range(int(v)):
            comp_data.append(data)
    comp_data = pd.DataFrame(comp_data)

    comp_mean = comp_data.mean(axis=0, numeric_only=True).to_dict()
    comp_std = comp_data.std(axis=0, numeric_only=True).to_dict()

    comp_mean = {"mean_" + str(col): value for col, value in comp_mean.items()}
    comp_std = {"std_" + str(col): value for col, value in comp_std.items()}

    # dist_mat = mol.distance_matrix
    descriptors = {}

    ###################### Modify this to add your own descriptors
    descriptors["metal_node_degree"] = len(nns)
    descriptors["nearest_neighbor_dist"] = min(dists)
    descriptors["average_dist"] = np.mean(dists)
    descriptors.update(element_featurizer(metal_species.symbol, "metal"))
    descriptors.update(
        element_featurizer(nearest.species.elements[0].symbol, "closest_neighbor")
    )
    descriptors.update(comp_mean)
    descriptors.update(comp_std)
    ######################
    feat_list.append(descriptors)

feats = pd.DataFrame(feat_list)
feats.to_csv("custom_features.csv")

y.to_csv("targets.csv")
