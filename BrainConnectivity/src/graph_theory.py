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
        tuple[pd.DataFrame]: Returns the `raw_data` DataFrame with appended
        columns
    """

    raw_data["Connectome"] = None
    raw_data["EVC"] = None
    raw_data["CLU"] = None
    raw_data["DIV"] = None
    for index, row in raw_data.iterrows():
        print(
            f"Row {index+1} of {raw_data.shape[0]} ({np.round((index/raw_data.shape[0])*100,2)}%)"
        )
        connectome = get_connectome(row["Data"])
        raw_data.at[index, "Connectome"] = connectome

        evc = get_evc(connectome)
        raw_data.at[index, "EVC"] = evc

        clu = get_clu(connectome)
        raw_data.at[index, "CLU"] = clu

        div = get_div(connectome)
        raw_data.at[index, "DIV"] = div

    return raw_data


def get_connectome(
    timeseries: np.ndarray, pearson: bool = True, fisher: bool = True
) -> float:
    """Returns the functional connectome based on pearson correlations.

    Args:
        pearson (bool, optional): Use Pearson Correlation. Defaults to True.
        Uses Pairwise correlation if False.
        fisher (bool, optional): Fisher Z-Project the resulting edges.
        Defaults to True.

    Returns:
        float: Connectome as a z by z float
    """
    # Remove any nodes which are 0 in any subject
    timeseries = np.delete(
        timeseries,
        [
            2,
            3,
            4,
            7,
            8,
            10,
            75,
            77,
            80,
            81,
            114,
            125,
            126,
            134,
            177,
            178,
            179,
            183,
            184,
            193,
            232,
            241,
            242,
            243,
            249,
            260,
            261,
            264,
            268,
            272,
            273,
            274,
            275,
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
    removed = np.where(np.sum(timeseries, axis=0) == 0)[0]
    timeseries = timeseries[:, np.sum(timeseries, axis=0) != 0]

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
        r_clipped = np.clip(connectome, -0.999999, 0.999999)
        connectome = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

    # Harmonization

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