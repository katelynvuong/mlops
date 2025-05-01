from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn

app = FastAPI(
    title="Wine Classifier (MLflow)",
    description="Predict wine class (0/1/2) using the MLflow-logged DecisionTree model",
    version="0.1",
)

class WineFeatures(BaseModel):
    # must match training feature order exactly
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

@app.on_event("startup")
def load_model():
    global model
    # point this to your actual run ID and artifact path:
    model_uri = "runs:/1d28cd866cb646ce92a20428fd1fd585/better_models"
    model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Classifier API"}

@app.post("/predict")
def predict_wine(sample: WineFeatures):
    # convert to 2D array in the same order used for training
    X = [[
        sample.alcohol,
        sample.malic_acid,
        sample.ash,
        sample.alcalinity_of_ash,
        sample.magnesium,
        sample.total_phenols,
        sample.flavanoids,
        sample.nonflavanoid_phenols,
        sample.proanthocyanins,
        sample.color_intensity,
        sample.hue,
        sample.od280_od315,
        sample.proline,
    ]]
    preds = model.predict(X)
    return {"prediction": int(preds[0])}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
