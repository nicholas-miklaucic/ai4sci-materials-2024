import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rho_plus as rp
from wat import wat
import functools as ft

import mendeleev

from pymatgen.core import Molecule
from pymatgen.io.xyz import XYZ
from pymatgen.analysis.local_env import JmolNN

from tqdm import tqdm

xyz = XYZ.from_file("tmQM_X_short.xyz")
y = pd.read_csv("tmQM_y_short.csv", sep=";")
y["eta"] = y.eval("(HOMO_Energy - LUMO_Energy) / 2")

# if you want dark mode, use is_dark=True instead
theme, cs = rp.mpl_setup(is_dark=True)

# context='talk' makes the text big so it's easy to see
# for your work, you may want to set this to 'notebook' instead
sns.set_context("notebook")


@ft.cache
def _element_featurizer(element):
    ele = mendeleev.element(element)
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
        features[f"ion_energy_{ion_energy}"] = ele.ionenergies.get(ion_energy, np.nan)

    return features


def element_featurizer(element, prefix=""):
    features = _element_featurizer(element)
    if prefix != "":
        features = {f"{prefix}_{k}": v for k, v in features.items()}

    return features


nn = JmolNN()
feat_list = []
for mol in tqdm(xyz.all_molecules[:], desc="Processing features..."):
    metal_i = max(range(len(mol.species)), key=lambda i: mol.species[i].Z)
    metal_site = mol.sites[metal_i]
    metal_species = mol.species[metal_i]
    nns = nn.get_nn(mol.get_boxed_structure(100, 100, 100), metal_i)
    dists = [np.linalg.norm(neighbor.coords - metal_site.coords) for neighbor in nns]
    nearest = nns[np.argmax(dists)]
    # dist_mat = mol.distance_matrix
    descriptors = {}
    descriptors["num_neighbors"] = len(nns)
    descriptors["nearest_neighbor_dist"] = min(dists)
    descriptors.update(element_featurizer(metal_species.symbol, "metal"))
    descriptors.update(
        element_featurizer(nearest.species.elements[0].symbol, "closest_neighbor")
    )
    feat_list.append(descriptors)

feats = pd.DataFrame(feat_list)

# print(feats.select_dtypes(exclude="number"))

if __name__ == "__main__":
    from sklearn.ensemble import (
        RandomForestRegressor,
        HistGradientBoostingRegressor,
        ExtraTreesRegressor,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.feature_selection import (
        VarianceThreshold,
        f_regression,
        SelectPercentile,
        r_regression,
        mutual_info_regression,
    )
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score

    # pcts = np.arange(10, 101, 10)
    # scores = []
    # for pct in pcts:
    pipe = make_pipeline(
        # SelectPercentile(r_regression, percentile=50),
        StandardScaler(),
        HistGradientBoostingRegressor(max_iter=200),
        # SGDRegressor(loss='huber', penalty='elasticnet')
        # KNeighborsRegressor(n_neighbors=20)
        # MLPRegressor(hidden_layer_sizes=(64,64), learning_rate='adaptive', activation='logistic',
        #              learning_rate_init=1e-3, solver='sgd', early_stopping=True, validation_fraction=0.1)
    )

    # scores.append(cross_val_score(pipe, feats, y['eta'], cv=5).mean())

    feat_name = "eta"
    # pipe.fit(feats, y[feat_name])
    # yhat = pipe.predict(feats)
    # print(r2_score(y[feat_name], yhat))
    print(cross_val_score(pipe, feats, y[feat_name][: len(feats)], cv=5).mean())
