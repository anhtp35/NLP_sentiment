"""FastAPI application for Emotion Classification API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import re

from deep_translator import GoogleTranslator
from model_utils import EmotionPredictor, EMOTION_NAMES


def is_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese characters."""
    vietnamese_pattern = r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]'
    return bool(re.search(vietnamese_pattern, text.lower()))


def translate_to_english(text: str) -> str:
    """Translate Vietnamese text to English using Google Translator."""
    try:
        if is_vietnamese(text):
            translator = GoogleTranslator(source='vi', target='en')
            return translator.translate(text)
        return text
    except Exception:
        return text  # Return original if translation fails

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Classification API",
    description="Multi-label emotion classification using DeBERTa + HEF features",
    version="1.0.0"
)

# Add CORS middleware for browser extension support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for extension compatibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


# ========================
# Pydantic Models
# ========================

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to classify")
    return_all: bool = Field(False, description="Return all emotion probabilities")


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")


class EmotionPrediction(BaseModel):
    emotion: str
    probability: float
    threshold: float
    detected: bool


class PredictionResponse(BaseModel):
    text: str
    predictions: List[EmotionPrediction]
    detected_emotions: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    emotions_supported: int


# ========================
# Startup Event
# ========================

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor
    
    # Support both direct paths and models/ subdirectory
    model_path = os.getenv("MODEL_PATH", "models/best_model.pt")
    thresholds_path = os.getenv("THRESHOLDS_PATH", "models/ensemble_thresholds.npy")
    
    # Fallback to root directory if models/ doesn't exist
    if not os.path.exists(model_path):
        model_path = "best_model.pt"
    if not os.path.exists(thresholds_path):
        thresholds_path = "ensemble_thresholds.npy"
    
    device = os.getenv("DEVICE", None)  # Auto-detect if not specified
    
    try:
        predictor = EmotionPredictor(
            model_path=model_path,
            thresholds_path=thresholds_path,
            device=device
        )
        print("✓ Model loaded and ready for inference!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


# ========================
# API Endpoints
# ========================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Emotion Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Classify single text",
            "/predict/batch": "POST - Classify multiple texts",
            "/health": "GET - Health check",
            "/emotions": "GET - List supported emotions"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor else "model_not_loaded",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else "unknown",
        "emotions_supported": len(EMOTION_NAMES)
    }


@app.get("/emotions", response_model=List[str])
async def list_emotions():
    """List all supported emotions."""
    return EMOTION_NAMES


# ========================
# Extension-Compatible Endpoint 
# ========================

class ExtensionTextRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict_for_extension(input_data: ExtensionTextRequest):
    """
    Predict emotions for a single text (Extension-compatible format).
    
    Returns format compatible with browser extension:
    {
        "original": "text",
        "translated": "text", 
        "emotions": [{"label": "joy", "score": 0.85}, ...]
    }
    
    Automatically detects Vietnamese and translates to English for better prediction.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    text = input_data.text
    if not text or len(text.strip()) == 0:
        return {"emotions": []}
    
    try:
        # Translate Vietnamese to English (model trained on English data)
        translated_text = translate_to_english(text)
        
        # Predict using translated text
        result = predictor.predict(translated_text, return_all=False)
        
        # Convert to extension-compatible format
        emotions = []
        for pred in result['predictions']:
            if pred['detected']:
                emotions.append({
                    "label": pred['emotion'],
                    "score": pred['probability']
                })
        
        return {
            "original": text,
            "translated": translated_text,
            "emotions": emotions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ========================
# Full API Endpoints 
# ========================

@app.post("/predict/full", response_model=PredictionResponse)
async def predict_full(input_data: TextInput):
    """
    Predict emotions with full details.
    
    - **text**: The text to classify (required)
    - **return_all**: If true, returns all emotion probabilities (default: false)
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = predictor.predict(input_data.text, return_all=input_data.return_all)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(input_data: BatchInput):
    """
    Predict emotions for multiple texts.
    
    - **texts**: List of texts to classify (max 100)
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = predictor.predict_batch(input_data.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ========================
# Run with: uvicorn app:app --reload
# ========================
