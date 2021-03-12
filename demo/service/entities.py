import typing as t
import typing_extensions as te

from pydantic import BaseModel, Field, ConstrainedFloat, PositiveFloat


class ModelInput(BaseModel):
    ms_subclass: str
    ms_zoning: str
    lotFrontage: float
    lotArea: float
    street: str
    alley: str
    lot_shape: str
    land_contour: str
    utilities: str
    lot_config: str
    land_slope: str
    neighborhood: str


# class ModelInput(BaseModel):
#     ms_subclass: str
#     ms_zoning: str
#     lotFrontage: PositiveFloat
#     lotArea: PositiveFloat
#     street: str
#     alley: str
#     lot_shape: str
#     land_contour: str
#     utilities: str
#     lot_config: str
#     land_slope: str
#     neighborhood: str


# MSSubClassType = te.Literal["20", "30", "40", "45", "50", "60", "70", "75", "80", "85", "90", "120", "150", "160", "180", "190"]
# MSZoningType = te.Literal["A", "C", "FV", "I", "RH", "RL", "RP", "RM"]
# Neighborhood = te.Literal["Blmgtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "Meadow", "Mitchel", "Names", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTwon", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"]

# class ModelInput(BaseModel):
#     ms_subclass: MSSubClassType
#     ms_zoning: MSZoningType
#     lotFrontage: PositiveFloat
#     lotArea: PositiveFloat
#     street: te.Literal["Grvl", "Pave"]
#     alley: te.Literal["Grvl", "Pave", "NA"]
#     lot_shape: te.Literal["Reg", "IR1", "IR2", "IR3"]
# 	land_contour: te.Literal["Lvl", "Bnk", "HLS", "Low"]
#     utilities: te.Literal["AllPub", "NoSewr", "NoSeWa", "ELO"]
#     lot_config: te.Literal["Inside", "Corner", "CulDSac", "FR2", "FR3"]
#     land_slope: te.Literal["Gtl", "Mod", "Sev"]
#     neighborhood: Neighborhood