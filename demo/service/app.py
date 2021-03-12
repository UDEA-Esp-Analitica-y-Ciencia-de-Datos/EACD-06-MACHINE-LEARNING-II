from fastapi import FastAPI, Depends  # type: ignore # noqa: E402
from pydantic import PositiveFloat

from service.entities import ModelInput

app = FastAPI(title="API to make inference with my great model", version="0.0.1")

@app.post("/", response_model=PositiveFloat)
async def make_prediction(input: ModelInput):
    raise NotImplementedError


@app.get("/")
async def service_status():
    """Check the status of the service"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
