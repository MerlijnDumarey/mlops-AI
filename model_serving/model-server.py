from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import joblib 

app = FastAPI()

MODEL_PATH = "/app/model/model.pkl" 

class PredictRequest(BaseModel):
    data: list

model = None

@app.on_event("startup")
async def load_model():
    """
    Loads the trained model when the FastAPI application starts up.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Make sure it's copied into the Docker image.")
    try:

        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.post("/v1/models/model:predict")
def predict(request: PredictRequest):
    """
    Prediction endpoint for the model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        input_data = np.array(request.data)

        prediction = model.predict(input_data).tolist()

        return {"predictions": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # this codeblock is just some local testing
    try:
        # simulate loading the model for local run
        if not os.path.exists(MODEL_PATH):
            print(f"[{MODEL_PATH}] not found. Creating a dummy model for local testing.")
            # create a dummy model for local testing if the actual model isn't present
            from sklearn.linear_model import LogisticRegression
            dummy_model = LogisticRegression()
            dummy_model.fit(np.array([[0,0],[1,1]]), np.array([0,1]))
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(dummy_model, MODEL_PATH)

        import asyncio
        asyncio.run(load_model())
        
        uvicorn.run(app, host="0.0.0.0", port=8501)
    except Exception as e:
        print(f"Error during local model server startup: {e}")
