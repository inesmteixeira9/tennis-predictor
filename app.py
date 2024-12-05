from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pipeline.predict import predict
from pipeline.models.logistic_regression import LogisticRegressionTrainer
from fastapi.templating import Jinja2Templates

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Create the FastAPI app
app = FastAPI()

# Configure CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home Route: Serves the index page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
async def trainRoute():
    """Route for training the model."""
    os.system("python main.py")  # This runs the model training process
    return JSONResponse({"message": "Training done successfully!"})


@app.post("/predict")
async def predictRoute(input: list):
    """Route for making predictions."""
    return predict(input)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
