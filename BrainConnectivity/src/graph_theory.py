import bct
import pandas as pd
import numpy as np
from IPython import display
from src.covMarket import covMarket


def append_connectome_data(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """Wrapper method to obtain the in-place modified data matrix containing
    connectome information in addition to the graph theory vectors.

    Args:
        raw_data (pd.DataFrame): Dataframe containing a `Data` column where
        each cell is a n by m matrix representing patient timeseries

    Returns:
        pd.DataFrame: Returns the `raw_data` DataFrame with appended
        columns
    """

    raw_data["Connectome"] = None
    for index, row in raw_data.iterrows():
        # print(
        #     f"Calculating Connectome for {row["Study"]} Subject {row["Subject_ID"]}..."
        # )
        connectome = get_connectome(row["Data"])
        raw_data.at[index, "Connectome"] = connectome

    return raw_data


def append_gt_data(raw_data: pd.DataFrame, harmonized: bool = True) -> pd.DataFrame:
    """Wrapper method to obtain the in-place modified data matrix containing
    connectome information in addition to the graph theory vectors.

    Args:
        raw_data (pd.DataFrame): Dataframe containing a `Data` column where
        each cell is a n by m matrix representing patient timeseries

        harmonized (bool): Flag to use 'Harmonized' column or 'Connectome'
        column for GT calculations

    Returns:
        pd.DataFrame: Returns the `raw_data` DataFrame with appended
        columns
    """

    raw_data["EVC"] = None
    raw_data["CLU"] = None
    raw_data["DIV"] = None
    for index, row in raw_data.iterrows():
        connectome = (
            raw_data.at[index, "Harmonized"]
            if harmonized
            else raw_data.at[index, "Connectome"]
        )

        if isinstance(connectome, np.ndarray):
            evc = get_evc(connectome)
            clu = get_clu(connectome)
            div = get_div(connectome)
        else:
            evc = None
            clu = None
            div = None

        raw_data.at[index, "EVC"] = evc
        raw_data.at[index, "CLU"] = clu
        raw_data.at[index, "DIV"] = div

    return raw_data


def get_connectome(
    timeseries: np.ndarray, pearson: bool = True, fisher: bool = True
) -> float:
    """Returns the unit normalized functional connectome based on pearson correlations.

    Args:
        pearson (bool, optional): Use Pearson Correlation. Defaults to True.
        Uses Pairwise correlation if False.
        fisher (bool, optional): Fisher Z-Project the resulting edges.
        Defaults to True.

    Returns:
        float: Connectome as a z by z float
    """
    # Remove any nodes which are 0 in any subject
    # Based on 2025-03-10 subjects, indices 3 4 8 9 81 180 181 198 249 are blank
    # and should be removed. In addition, the following nodes are statically
    # removed spheres: 82 127 184 185 233 250 273 274 277 280 281 284 289 290
    # 293 294 (index+1).
    print("INDEX TO REMOVE: ", np.where(np.sum(timeseries, axis=0) == 0)[0])
    timeseries = np.delete(
        timeseries,
        [
            3,
            4,
            8,
            9,
            180,
            181,
            198,
            81,
            126,
            183,
            184,
            232,
            249,
            272,
            273,
            276,
            279,
            280,
            283,
            288,
            289,
            292,
            293,
        ],
        axis=1,
    )

    to_remove = np.where(np.sum(timeseries, axis=0) == 0)[0]
    if len(to_remove) > 0:
        print("NOT REMOVED: ", to_remove)

    # timeseries = timeseries[:, np.sum(timeseries, axis=0) != 0]

    # Normalize each column to have unit Euclidean length
    timeseries = timeseries / np.linalg.norm(timeseries, axis=0)

    # Obtain correlation matrix
    correlation = timeseries.T @ timeseries

    # Pearson if true, otherwise pairwise
    connectome = correlation
    if pearson:
        shrunk = covMarket(pd.DataFrame(correlation))
        # condition = np.linalg.cond(shrunk)
        # rank = np.linalg.matrix_rank(shrunk)
        inverse = np.linalg.inv(shrunk)
        sqrt_diag = np.sqrt(np.diag(inverse))
        connectome = inverse / np.outer(sqrt_diag, sqrt_diag)
    connectome = connectome - np.diag(np.diag(connectome))  # Remove diag

    # Fisher z-projection
    if fisher:
        connectome = np.multiply(
            np.subtract(np.arctanh(connectome), np.arctanh(0)),
            np.sqrt(connectome.shape[0] - 0 - 3),
        )

    return connectome


def get_evc(connectome: float) -> float:
    """Returns the eigenvector corresponding to the largest eigenvalue of the
    provided connectome

    Args:
        connectome (float): Connectomic data

    Returns:
        float: Eigenvector corresponding to the largest eigenvalue of
        connectome
    """
    # Will reduce dimensions and only return a vector
    D, V = np.linalg.eig(connectome)
    sorted_indices = np.argsort(D)
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    V = np.abs(V)
    return V[:, -1]


def get_div(connectome: float) -> float:
    # Will reduce dimensions and only return a vector
    cav, q = bct.community_louvain(connectome, B="negative_sym")
    dpos, dneg = bct.diversity_coef_sign(connectome, cav)
    return dpos


def get_clu(connectome: float) -> float:
    # Will reduce dimensions and only return a vector
    cpos, cneg = bct.clustering_coef_wu_sign(connectome)
    return cpos
