import typing as t

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_estimator(hyperparams: t.Dict[str, t.Any]):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for name, params in hyperparams.items():
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "regressor": RandomForestRegressor,
        "age-extractor": AgeExtractor,
        "column-transformer": CustomColumnTransformer,
        "simplified-transformer": SimplifiedTransformer,
    }


class AgeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["RemodAddAge"] = X["YrSold"] - X["YearRemodAdd"]
        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
        return X


class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    _categorical_columns = (
        "MSSubClass,MSZoning,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,"
        + "Neighborhood,Condition1,Condition2,BldgType,HouseStyle,RoofStyle,RoofMatl,"
        + "Exterior1st,MasVnrType,Foundation,Heating,Electrical,GarageType,PavedDrive,"
        + "MiscFeature,SaleType,SaleCondition,OverallQual,OverallCond,ExterQual,"
        + "ExterCond,BsmtQual,BsmtCond,BsmtFinType1,HeatingQC,PoolQC,Fence,KitchenQual,"
        + "Functional,FireplaceQu,GarageFinish,GarageQual,GarageCond,BsmtExposure,"
        + "BsmtFinType2,Exterior2nd,MoSold"
    ).split(",")

    _binary_columns = "Street,CentralAir".split(",")

    _float_columns = (
        "LotFrontage,LotArea,MasVnrArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,1stFlrSF,"
        + "2ndFlrSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,MiscVal,LowQualFinSF,"
        + "GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,"
        + "TotRmsAbvGrd,Fireplaces,GarageCars,GarageArea,WoodDeckSF,OpenPorchSF,"
        + "HouseAge,RemodAddAge,GarageAge"
    ).split(",")

    _ignored_columns = "YrSold,YearBuilt,YearRemodAdd,GarageYrBlt".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("droper", "drop", type(self)._ignored_columns),
                ("binarizer", OrdinalEncoder(), type(self)._binary_columns),
                (
                    "one_hot_encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    type(self)._categorical_columns,
                ),
                ("scaler", StandardScaler(), type(self)._float_columns),
            ],
            remainder="drop",
        )

    def fit(self, X, y=None):
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        return self._column_transformer.transform(X)


class SimplifiedTransformer(BaseEstimator, TransformerMixin):
    """This is just for easy of demonstration"""

    _columns_to_keep = "HouseAge,GarageAge,LotArea,Neighborhood,HouseStyle".split(",")

    def __init__(self):
        self._column_transformer = ColumnTransformer(
            transformers=[
                ("binarizer", OrdinalEncoder(), ["Neighborhood", "HouseStyle"]),
            ],
            remainder="drop",
        )

    def fit(self, X, y=None):
        columns = type(self)._columns_to_keep
        X_ = X[columns]
        self._column_transformer = self._column_transformer.fit(X_, y=y)
        return self

    def transform(self, X):
        columns = type(self)._columns_to_keep
        X_ = X[columns]
        X_ = self._column_transformer.transform(X_)
        return X_
