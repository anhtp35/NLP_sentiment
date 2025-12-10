"""Model utilities for emotion classification inference."""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji

# Constants
BACKBONE = "microsoft/deberta-v3-large"
MAX_LEN = 128
HEF_DIM = 13
NUM_LABELS = 28

EMOTION_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
    'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]


class HybridClassifier(nn.Module):
    """Hybrid classifier with transformer backbone and HEF features."""
    
    def __init__(self, backbone, hef_dim, hidden_size, num_labels, proj=512, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size + hef_dim, proj),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(proj, num_labels)
        )
    
    def forward(self, input_ids=None, attention_mask=None, hef=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.last_hidden_state[:, 0, :]  # CLS-like pooling
        if hef is None:
            hef = torch.zeros((pooled.size(0), HEF_DIM), device=pooled.device)
        x = torch.cat([pooled, hef], dim=-1)
        logits = self.head(x)
        return logits


class EmotionPredictor:
    """Main class for emotion prediction inference."""
    
    def __init__(self, model_path: str, thresholds_path: str, device: str = None):
        """
        Initialize the emotion predictor.
        
        Args:
            model_path: Path to the saved model weights (.pt file)
            thresholds_path: Path to the thresholds (.npy file)
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        
        # Load backbone and create model
        print("Loading backbone model...")
        backbone = AutoModel.from_pretrained(BACKBONE)
        hidden_size = backbone.config.hidden_size
        
        self.model = HybridClassifier(
            backbone=backbone,
            hef_dim=HEF_DIM,
            hidden_size=hidden_size,
            num_labels=NUM_LABELS
        ).to(self.device)
        
        # Load trained weights
        print(f"Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load thresholds
        print(f"Loading thresholds from: {thresholds_path}")
        self.thresholds = np.load(thresholds_path)
        
        # Load HEF tools
        print("Loading HEF tools (spaCy, VADER)...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.vader = SentimentIntensityAnalyzer()
        
        print("✓ Model loaded successfully!")
    
    def extract_hef_features(self, text: str) -> np.ndarray:
        """Extract Hand-crafted Emotion Features from text."""
        doc = self.nlp(text)
        
        # POS counts
        pos_counts = {'ADJ': 0, 'ADV': 0, 'INTJ': 0, 'VERB': 0, 'NOUN': 0}
        for tok in doc:
            if tok.pos_ in pos_counts:
                pos_counts[tok.pos_] += 1
        
        # Punctuation and style
        exclam = text.count('!')
        qmarks = text.count('?')
        allcaps = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        
        # VADER sentiment
        vs = self.vader.polarity_scores(text)
        
        # Emoji count
        emojis = sum(1 for c in text if c in emoji.EMOJI_DATA)
        
        feat = [
            pos_counts['ADJ'], pos_counts['ADV'], pos_counts['INTJ'],
            pos_counts['VERB'], pos_counts['NOUN'],
            exclam, qmarks, allcaps,
            vs['neg'], vs['neu'], vs['pos'], vs['compound'],
            emojis
        ]
        
        return np.array(feat, dtype=np.float32)
    
    def predict(self, text: str, return_all: bool = False) -> dict:
        """
        Predict emotions for a given text.
        
        Args:
            text: Input text to classify
            return_all: If True, return probabilities for all emotions.
                       If False, only return emotions above threshold.
        
        Returns:
            Dictionary with predictions
        """
        # Tokenize
        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        
        # Extract HEF features
        hef_np = self.extract_hef_features(text)
        hef_tensor = torch.from_numpy(hef_np).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                hef=hef_tensor
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get predictions above threshold
        predictions = []
        for i in range(NUM_LABELS):
            if return_all or probs[i] >= self.thresholds[i]:
                predictions.append({
                    'emotion': EMOTION_NAMES[i],
                    'probability': float(probs[i]),
                    'threshold': float(self.thresholds[i]),
                    'detected': bool(probs[i] >= self.thresholds[i])
                })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'text': text,
            'predictions': predictions,
            'detected_emotions': [p['emotion'] for p in predictions if p['detected']]
        }
    
    def predict_batch(self, texts: list) -> list:
        """Predict emotions for a batch of texts."""
        return [self.predict(text) for text in texts]
