from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from app.libs import data_utils 
from datetime import date
from app.config import APP_NAME, APP_VERSION, API_PREFIX, CONFIG, DOCS
from app import build_features
from app.models import random_forest

# Initialize FastAPI app
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.root_path = f'/{APP_NAME}{API_PREFIX}'
app.openapi_url = '/openapi.json'

raw_data_path = CONFIG['app']['raw_data_path']
interim_data = CONFIG['app']['interim_data']

class RawData(BaseModel):
    date                    : date
    winner                  : str
    loser                   : str
    wrank                   : int
    lrank                   : int
    b365w                   : float
    b365l                   : float
    surface                 : str

class InputData(BaseModel):
    rank_p1                 : int
    rank_p2                 : int
    rank_diff               : int
    rank_ratio              : float
    odd_diff                : float
    odd_ratio               : float
    surface_Clay            : int
    surface_Hard            : int
    surface_Grass           : int
    consecutive_wins_p1     : int 
    consecutive_wins_p2     : int
    consecutive_losses_p1   : int
    consecutive_losses_p2   : int
    days_last_win_p1        : int
    days_last_win_p2        : int    


# upload data file endpoint
@app.post('/extract/csvfiles', tags=['Extraction'], description=DOCS['csvfiles']['description'])
def upload_csvfile(csv_file: UploadFile = File(...)):
    data_utils.upload_csvfile(csv_file, raw_data_path)


# train data endpoint
@app.get("/get_features", status_code=200)
def get_features():
    
    build_features.transform_data()

    df, features = build_features.add_ranks(df, features)

    # Add the difference and ratio between players
    df, features = build_features.add_rank_dif(df, features)
    df, features = build_features.add_rank_ratio(df, features)

    # Add the difference and ratio between players
    df, features = build_features.add_odd_dif(df, features)
    df, features = build_features.add_odd_ratio(df, features)

    # Add a binary column for each surface
    df, features = build_features.OHE_surface(df, features)

    # Add consecutive wins or losses
    df, features = build_features.add_consecutive_wins_and_losses(df, features)


# train data endpoint
@app.get("/train", status_code=200)
def train(data: InputData):
    random_forest.train_data(interim_data)


# predict endpoint
@app.get("/predict", status_code=200)
def predict(data: InputData):
    prediction = random_forest.predict_data(interim_data)
    return {"prediction": prediction[0]}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)