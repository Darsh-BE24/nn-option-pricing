"""Parse an NSE option-chain CSV (the 'Download (.csv)' button on the option-chain page)."""
from __future__ import annotations
import numpy as np
import pandas as pd

CALL_COLS = ["c_blank", "c_oi", "c_chg_oi", "c_vol", "c_iv", "c_ltp", "c_chg",
             "c_bid_qty", "c_bid", "c_ask", "c_ask_qty", "strike"]
PUT_COLS = ["p_bid_qty", "p_bid", "p_ask", "p_ask_qty", "p_chg", "p_ltp",
            "p_iv", "p_vol", "p_chg_oi", "p_oi", "p_blank"]


def _num(s: pd.Series) -> pd.Series:
    """'1,690' -> 1690.0 ; '-' -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip()
                         .replace({"-": np.nan, "": np.nan, "nan": np.nan}), errors="coerce")


def load_chain(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=2, header=None, dtype=str)
    raw = raw.iloc[:, : len(CALL_COLS) + len(PUT_COLS)]
    raw.columns = CALL_COLS + PUT_COLS
    df = raw.drop(columns=["c_blank", "p_blank"]).apply(_num)
    df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    return df
