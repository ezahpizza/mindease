# Standard Library Imports
import os
import json
import jwt
import uvicorn
from datetime import datetime, timedelta
from typing import List, Optional

# Third-Party Imports
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from huggingface_hub import login
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
from bson import ObjectId

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
    allow_origins=["*"],  
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

class ChatMessage(BaseModel):
    content: str
    timestamp: datetime = Field(default_factory=datetime.timezone.utc)
    
class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime = Field(default_factory=datetime.timezone.utc)

# Add this class for managing the chatbot
class TherapyChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "ezahpizza/mindease_chatbot"
        self.tokenizer = None
        self.model = None
        self.hf_token = "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"
        
    async def load_model(self):
        if self.model is None:
            try:
                if not self.hf_token:
                    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
                
                # Initialize tokenizer with auth
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token,
                    revision="main"
                )
                
                # Initialize model with auth
                self.model = GPT2LMHeadModel.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token,
                    revision="main"
                )
                
                # Configure special tokens
                special_tokens = {
                    'additional_special_tokens': ['<|context|>', '<|response|>']
                }
                self.tokenizer.add_special_tokens(special_tokens)
                self.model.resize_token_embeddings(len(self.tokenizer))
                
                # Configure padding
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
                self.model.to(self.device)
                self.model.eval()
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise
    
    async def generate_response(self, message: str, max_length: int = 100) -> str:
        await self.load_model()
        
        input_text = f"<|context|>{message}<|response|>"
        
        try:
            # Encode with attention mask
            encoded = self.tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=2,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

            # Extract response part
            response = generated_text.split("<|response|>")[-1].strip()
            # Remove any remaining special tokens
            response = response.replace(self.tokenizer.eos_token, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now."

# Initialize chatbot instance
chatbot = TherapyChatbot()

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

# Chat API endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_input: ChatInput,
    current_user: User = Depends(get_current_user)
):
    try:
        # Generate response
        response = await chatbot.generate_response(chat_input.message)
        
        # Create chat message documents
        user_message = {
            "user_id": str(current_user.id),
            "content": chat_input.message,
            "type": "user",
            "timestamp": datetime.utcnow()
        }
        
        bot_message = {
            "user_id": str(current_user.id),
            "content": response,
            "type": "bot",
            "timestamp": datetime.utcnow()
        }
        
        # Store messages in database
        await database.chat_messages.insert_many([user_message, bot_message])
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}"
        )

@app.get("/chat/history")
async def get_chat_history(
    current_user: User = Depends(get_current_user)
):
    try:
        messages = await database.chat_messages.find(
            {"user_id": str(current_user.id)}
        ).sort("timestamp", 1).to_list(length=100)
        
        return jsonable_encoder_custom(messages)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chat history: {str(e)}"
        )

# Startup and shutdown events
# In your main.py startup
@app.on_event("startup")
async def startup_db_client():
    # Initialize MongoDB
    app.mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    app.mongodb = app.mongodb_client.mental_health_db
    
    # Initialize Hugging Face authentication
    hf_token = "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"
    if hf_token:
        login(hf_token)
        print("Successfully authenticated with Hugging Face Hub")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)