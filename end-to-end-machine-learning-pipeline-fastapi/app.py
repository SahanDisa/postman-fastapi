import models.ml.classifier as clf
from fastapi import FastAPI, Body
from joblib import load
from models.iris import Iris
import uvicorn

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")

@app.get("/", tags=["Welcome PostStudents"])
def read_root():
    return {"Hello": "World"}

@app.on_event('startup')
async def load_model():
    clf.model = load('models/ml/iris_dt_v1.joblib')

@app.post('/predict', tags=["predictions"])
async def get_prediction(iris: Iris):
    data = dict(iris)['data']
    prediction = clf.model.predict(data).tolist()
    log_proba = clf.model.predict_proba(data).tolist()
    return {"prediction": prediction, "log_proba": log_proba}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# --------- Testing-------- 
# {
#   "data": [
#     [4.8,3,1.4,0.3],
#     [2,1,3.2,1.1]
#   ]
# }


# curl -X 'POST' \
#   'http://127.0.0.1:5000/predict' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "data": [
#     [4.8,3,1.4,0.3],
#     [2,1,3.2,1.1]
#   ]
# }'

#docker build -t iris-ml-build .
#docker run -d -p 80:80 --name iris-api iris-ml-build