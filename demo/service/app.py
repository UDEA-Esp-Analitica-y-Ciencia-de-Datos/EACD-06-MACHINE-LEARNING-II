import sys
import typing as t
from datetime import datetime
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, Body  # type: ignore # noqa: E402
from pydantic import BaseSettings, PositiveFloat

from entities import ModelInput

app = FastAPI(title="API to make inference with my great model", version="0.0.1")


class Settings(BaseSettings):
    serialized_model_path: str
    model_lib_dir: str


@lru_cache(None)
def get_settings():
    return Settings()


@lru_cache(None)
def load_estimator():
    sys.path.append(get_settings().model_lib_dir)
    estimator = joblib.load(get_settings().serialized_model_path)
    return estimator


# @app.post("/")
# async def make_prediction(input_: str = Body(...), estimator=Depends(load_estimator)):
#     """
#     Call this using something like

#     YrSold=2010,YearBuilt=1970,YearRemodAdd=1999,GarageYrBlt=1980,LotArea=24,Neighborhood=Blmngtn,HouseStyle=SFoyer
#     """
#     input_dict = {}
#     for var in input_.split(","):
#         name, value = var.split("=")
#         if value.isnumeric():

#             value = int(value)
#         input_dict[name] = [value]
#     X = pd.DataFrame(input_dict)
#     prediction = estimator.predict(X).tolist()
#     return prediction


# @app.post("/")
# async def make_prediction(
#     inputs: t.List[ModelInput] = Body(...),
#     estimator=Depends(load_estimator),
# ):
#     X = pd.DataFrame([row.dict() for row in inputs])
#     prediction = estimator.predict(X).tolist()
#     return prediction


# @app.post("/", response_model=t.List[float])
# async def make_prediction(
#     inputs: t.List[ModelInput] = Body(...),
#     estimator=Depends(load_estimator),
# ):
#     X = pd.DataFrame([row.dict() for row in inputs])
#     prediction = estimator.predict(X).tolist()
#     return prediction


class Logger:
    def __init__(self, file: t.TextIO = sys.stdout):
        self.file = file

    def log(self, inputs: t.List[ModelInput]):
        for row in inputs:
            record = {"datetime": datetime.now(), "input": row.dict()}
            print(record, file=self.file)


def get_logger():
    return Logger()


@app.post("/", response_model=t.List[float])
async def make_prediction(
    inputs: t.List[ModelInput] = Body(...),
    estimator=Depends(load_estimator),
    logger=Depends(get_logger),
):
    logger.log(inputs)
    X = pd.DataFrame([row.dict() for row in inputs])
    prediction = estimator.predict(X).tolist()
    return prediction


@app.get("/")
async def service_status():
    """Check the status of the service"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
