"""Graph topology feature engineering for node (account) attributes.

The feature table is keyed by the raw account id and carries, for every
account, a rich behavioural+topological profile:

* ``in_degree`` / ``out_degree`` — transaction counts receiving/sending.
* ``mean_amount_sent`` / ``mean_amount_received`` — monetary size profile.
* ``total_amount_sent`` / ``total_amount_received`` — flow magnitude.
* ``sent_velocity`` / ``received_velocity`` — transactions in the last
  window (see ``velocity_window_seconds``).
* ``mean_inter_tx_seconds`` — average time between this account's own
  outbound transactions (transactional spacing).
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "mean_amount_sent",
    "mean_amount_received",
    "total_amount_sent",
    "total_amount_received",
    "sent_velocity",
    "received_velocity",
    "mean_inter_tx_seconds",
]


def compute_node_features(
    df: pd.DataFrame, velocity_window_seconds: float = 86400.0
) -> pd.DataFrame:
    """Compute per-account topology/behavioural features.

    Args:
        df: Canonical transaction table (see :mod:`.ingestion`).
        velocity_window_seconds: Recency window, measured from the maximal
            timestamp in ``df``, used for the velocity features.

    Returns:
        DataFrame indexed by the raw account id with
        :data:`FEATURE_COLUMNS` as columns. Accounts absent from the edge
        list are guaranteed to appear via the union of ``src``/``dst``.
    """
    if df.empty:
        return pd.DataFrame(columns=["account", *FEATURE_COLUMNS]).set_index("account")

    out_agg = df.groupby("src")["amount"].agg(
        out_degree="count",
        mean_amount_sent="mean",
        total_amount_sent="sum",
    )
    in_agg = df.groupby("dst")["amount"].agg(
        in_degree="count",
        mean_amount_received="mean",
        total_amount_received="sum",
    )

    max_ts = df["timestamp"].max()
    recent = df[df["timestamp"] >= max_ts - velocity_window_seconds]
    sent_velocity = recent.groupby("src").size().rename("sent_velocity")
    received_velocity = recent.groupby("dst").size().rename("received_velocity")

    # Mean spacing between consecutive outbound transactions.
    sorted_out = df.sort_values("timestamp")
    gaps = sorted_out.groupby("src")["timestamp"].diff().dropna()
    inter_tx = gaps.groupby(sorted_out["src"]).mean().rename("mean_inter_tx_seconds")

    accounts = pd.Index(sorted(set(df["src"]).union(set(df["dst"]))), name="account")
    features = pd.DataFrame(index=accounts)
    for frame in (out_agg, in_agg, sent_velocity, received_velocity, inter_tx):
        features = features.join(frame.to_frame() if isinstance(frame, pd.Series) else frame)

    fill_defaults = {
        "in_degree": 0,
        "out_degree": 0,
        "sent_velocity": 0,
        "received_velocity": 0,
        "mean_inter_tx_seconds": np.nan,  # unknown → imputed downstream
    }
    for column, default in fill_defaults.items():
        features[column] = features[column].fillna(default)

    features[["mean_amount_sent", "total_amount_sent"]] = features[
        ["mean_amount_sent", "total_amount_sent"]
    ].fillna(0.0)
    features[["mean_amount_received", "total_amount_received"]] = features[
        ["mean_amount_received", "total_amount_received"]
    ].fillna(0.0)

    # Accounts with a single outbound tx have no gaps; use the global mean.
    global_mean_gap = float(df["timestamp"].max() - df["timestamp"].min()) or 1.0
    features["mean_inter_tx_seconds"] = features["mean_inter_tx_seconds"].fillna(global_mean_gap)

    # A laundering flag propagates to every account touching a dirty edge.
    dirty = df[df["is_laundering"] == 1]
    label_accounts = pd.Index(sorted(set(dirty["src"]).union(set(dirty["dst"]))), name="account")
    features["label"] = 0
    features.loc[label_accounts, "label"] = 1
    return features
