import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset(df)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "Id"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _fix_pool_quality,
            _fix_misc_feature,
            _fix_fireplace_quality,
            _fix_garage_variables,
            _fix_lot_frontage,
            _fix_alley,
            _fix_fence,
            _fix_masvnr_variables,
            _fix_electrical,
            _fix_basement_variables,
            _fix_unhandled_nulls,
        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def _fix_pool_quality(df):
    num_total_nulls = df["PoolQC"].isna().sum()
    num_nulls_when_poolarea_is_zero = df[df["PoolArea"] == 0]["PoolQC"].isna().sum()
    assert num_nulls_when_poolarea_is_zero == num_total_nulls
    num_nulls_when_poolarea_is_not_zero = df[df["PoolArea"] != 0]["PoolQC"].isna().sum()
    assert num_nulls_when_poolarea_is_not_zero == 0
    df["PoolQC"] = df["PoolQC"].fillna("NP")
    return df


def _fix_misc_feature(df):
    num_total_nulls = df["MiscFeature"].isna().sum()
    num_nulls_when_miscval_is_zero = df[df["MiscVal"] == 0]["MiscFeature"].isna().sum()
    num_nulls_when_miscval_is_not_zero = (
        df[df["MiscVal"] != 0]["MiscFeature"].isna().sum()
    )
    assert num_nulls_when_miscval_is_zero == num_total_nulls
    assert num_nulls_when_miscval_is_not_zero == 0
    df["MiscFeature"] = df["MiscFeature"].fillna("No MF")
    return df


def _fix_fireplace_quality(df):
    num_total_nulls = df["FireplaceQu"].isna().sum()
    num_nulls_when_fireplaces_is_zero = (
        df[df["Fireplaces"] == 0]["FireplaceQu"].isna().sum()
    )
    num_nulls_when_fireplaces_is_not_zero = (
        df[df["Fireplaces"] != 0]["FireplaceQu"].isna().sum()
    )
    assert num_nulls_when_fireplaces_is_zero == num_total_nulls
    assert num_nulls_when_fireplaces_is_not_zero == 0
    df["FireplaceQu"] = df["FireplaceQu"].fillna("No FP")
    return df


def _fix_garage_variables(df):
    num_area_zeros = (df["GarageArea"] == 0).sum()
    num_cars_zeros = (df["GarageCars"] == 0).sum()
    num_both_zeros = ((df["GarageArea"] == 0) & (df["GarageCars"] == 0.0)).sum()
    assert num_both_zeros == num_area_zeros == num_cars_zeros
    for colname in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        num_total_nulls = df[colname].isna().sum()
        num_nulls_when_area_and_cars_capacity_is_zero = (
            df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)][colname]
            .isna()
            .sum()
        )
        num_nulls_when_area_and_cars_capacity_is_not_zero = (
            df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)][colname]
            .isna()
            .sum()
        )
        assert num_total_nulls == num_nulls_when_area_and_cars_capacity_is_zero
        assert num_nulls_when_area_and_cars_capacity_is_not_zero == 0
        df[colname] = df[colname].fillna("No Ga")

    num_total_nulls = df["GarageYrBlt"].isna().sum()
    num_nulls_when_area_and_cars_is_zero = (
        df[(df["GarageArea"] == 0.0) & (df["GarageCars"] == 0.0)]["GarageYrBlt"]
        .isna()
        .sum()
    )
    num_nulls_when_area_and_cars_is_not_zero = (
        df[(df["GarageArea"] != 0.0) & (df["GarageCars"] != 0.0)]["GarageYrBlt"]
        .isna()
        .sum()
    )
    assert num_nulls_when_area_and_cars_is_zero == num_total_nulls
    assert num_nulls_when_area_and_cars_is_not_zero == 0
    df["GarageYrBlt"].where(
        ~df["GarageYrBlt"].isna(), other=df["YrSold"] + 1, inplace=True
    )

    return df


def _fix_lot_frontage(df):
    assert (df["LotFrontage"] == 0).sum() == 0
    df["LotFrontage"].fillna(0, inplace=True)
    return df


def _fix_alley(df):
    df["Alley"].fillna("NA", inplace=True)
    return df


def _fix_fence(df):
    df["Fence"].fillna("NF", inplace=True)
    return df


def _fix_masvnr_variables(df):
    df = df.dropna(subset=["MasVnrType", "MasVnrArea"])
    df = df[~((df["MasVnrType"] == "None") & (df["MasVnrArea"] != 0.0))]
    return df


def _fix_electrical(df):
    df.dropna(subset=["Electrical"], inplace=True)
    return df


def _fix_basement_variables(df):
    colnames = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    cond = ~(
        df["BsmtQual"].isna()
        & df["BsmtCond"].isna()
        & df["BsmtExposure"].isna()
        & df["BsmtFinType1"].isna()
        & df["BsmtFinType2"].isna()
    )
    for c in colnames:
        df[c].where(cond, other="NB", inplace=True)
    return df


def _fix_unhandled_nulls(df):
    df.dropna(inplace=True)
    return df
