# Required imports
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional
import xgboost as xgb
import pandas as pd
import numpy as np
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import json
import os
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime
# Load environment variables
load_dotenv()

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def jsonable_encoder_custom(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: jsonable_encoder_custom(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable_encoder_custom(i) for i in obj]
    return obj

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    json_encoder=JSONEncoder  # Add this line
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "nakapoeskibidimentalhealth")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
db_client = AsyncIOMotorClient(MONGODB_URL)
database = db_client.mental_health_db

# Input data models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
        
    @classmethod
    def from_mongo(cls, data: dict):
        if not data:
            return None
        if "_id" in data:
            data["id"] = str(data["_id"])
            del data["_id"]
        return cls(**data)

class PredictionInput(BaseModel):
    gender: int
    age: float
    city: int
    profession: int
    academic_pressure: float
    work_pressure: float
    cgpa: float
    study_satisfaction: float
    job_satisfaction: float
    sleep_duration: int
    dietary_habits: int
    degree: int
    suicidal_thoughts: int
    work_study_hours: float
    financial_stress: float
    mi_history: int

class PredictionResult(BaseModel):
    prediction: float
    prediction_label: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

import os

# Get the absolute path to the directory containing main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the absolute path to model.json
MODEL_PATH = os.path.join(BASE_DIR, 'model.json')

# Load the model using the absolute path
try:
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print("Model feature names:", model.feature_names)
except xgb.core.XGBoostError as e:
    print(f"Error loading model from {MODEL_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model directory: {BASE_DIR}")
    raise Exception(f"Failed to load XGBoost model: {str(e)}")

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    
    user_doc = await database.users.find_one({"username": username})
    if user_doc is None:
        raise credentials_exception
    return User.from_mongo(user_doc)  

# API endpoints
@app.post("/register")
async def register_user(user: UserCreate):
    # Check if user already exists
    if await database.users.find_one({"username": user.username}):
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed_password
    user_dict["created_at"] = datetime.utcnow()
    
    result = await database.users.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)  # Convert ObjectId to string
    
    return {"message": "User registered successfully"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await database.users.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(
        data={"sub": user["username"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    current_user: User = Depends(get_current_user)
):
    # Create a DataFrame with the correct feature names
    input_dict = {
        'gender': [input_data.gender],
        'age': [input_data.age],
        'city': [input_data.city],
        'profession': [input_data.profession],
        'academic_pressure': [input_data.academic_pressure],
        'work_pressure': [input_data.work_pressure],
        'cgpa': [input_data.cgpa],
        'study_satisfaction': [input_data.study_satisfaction],
        'job_satisfaction': [input_data.job_satisfaction],
        'sleep_duration': [input_data.sleep_duration],
        'dietary_habits': [input_data.dietary_habits],
        'degree': [input_data.degree],
        'suicidal_thoughts': [input_data.suicidal_thoughts],
        'work_study_hours': [input_data.work_study_hours],
        'financial_stress': [input_data.financial_stress],
        'mi_history': [input_data.mi_history]
    }

    # Convert to DataFrame to ensure feature order matches training data
    import pandas as pd
    input_df = pd.DataFrame(input_dict)
    
    # Convert DataFrame to DMatrix with feature names
    dmatrix = xgb.DMatrix(input_df)
    
    try:
        # Make prediction
        prediction = model.predict(dmatrix)[0]
        prediction_label = "High risk of depression" if prediction > 0.5 else "Low risk of depression"
        
        # Create prediction result
        result = PredictionResult(
            prediction=float(prediction),
            prediction_label=prediction_label
        )
        
        # Store prediction in database
        prediction_doc = {
            "user_id": str(current_user.id),  # Ensure it's a string
            "input_data": jsonable_encoder_custom(input_data.dict()),
            "prediction": float(result.prediction),  # Ensure it's a float
            "prediction_label": result.prediction_label,
            "timestamp": datetime.utcnow()  # This will be converted by our encoder
        }
    
        await database.predictions.insert_one(prediction_doc)
        return jsonable_encoder_custom(result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/user/predictions")
async def get_user_predictions(current_user: User = Depends(get_current_user)):
    predictions = await database.predictions.find(
        {"user_id": str(current_user.id)}
    ).to_list(length=100)
    
    predictions = jsonable_encoder_custom(predictions)
    return predictions

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    app.mongodb = app.mongodb_client.mental_health_db

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)