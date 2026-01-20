"""
Model persistence utilities for loading and saving trained ML models.
"""

import pickle
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Default trained models directory
TRAINED_DIR = Path(__file__).parent / "trained"


class ModelRegistry:
    """
    Registry for managing trained ML models.
    Handles loading, caching, and serving models.
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    _metadata: Dict[str, Dict] = {}
    
    def __new__(cls):
        """Singleton pattern - only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._metadata = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        """Get the singleton instance."""
        return cls()
    
    def load_model(self, name: str, force_reload: bool = False) -> Optional[Any]:
        """
        Load a trained model by name.
        
        Args:
            name: Model name (e.g., 'forecaster', 'clustering')
            force_reload: If True, reload from disk even if cached
            
        Returns:
            Loaded model or None if not found
        """
        # Check cache first
        if not force_reload and name in self._models:
            return self._models[name]
        
        # Try to load from disk
        model_path = TRAINED_DIR / f"{name}.pkl"
        
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self._models[name] = model
            
            # Load metadata if available
            meta_path = TRAINED_DIR / f"{name}_metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self._metadata[name] = json.load(f)
            
            print(f"✅ Loaded model: {name}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            return None
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all available trained models."""
        model_names = [
            'forecaster',
            'capacity_planning',
            'underserved_scoring',
            'fraud_detector',
            'clustering',
            'hotspot_detector',
            'cohort_model'
        ]
        
        loaded = {}
        for name in model_names:
            model = self.load_model(name)
            if model is not None:
                loaded[name] = model
        
        return loaded
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a model from cache or load it."""
        if name in self._models:
            return self._models[name]
        return self.load_model(name)
    
    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a model."""
        if name not in self._metadata:
            meta_path = TRAINED_DIR / f"{name}_metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self._metadata[name] = json.load(f)
        return self._metadata.get(name)
    
    def is_trained(self, name: str) -> bool:
        """Check if a model has been trained."""
        model_path = TRAINED_DIR / f"{name}.pkl"
        return model_path.exists()
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get training status for all models."""
        model_names = [
            'forecaster',
            'capacity_planning',
            'underserved_scoring',
            'fraud_detector',
            'clustering',
            'hotspot_detector',
            'cohort_model'
        ]
        
        status = {}
        for name in model_names:
            model_path = TRAINED_DIR / f"{name}.pkl"
            meta_path = TRAINED_DIR / f"{name}_metadata.json"
            
            if model_path.exists():
                stat = model_path.stat()
                metadata = {}
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                
                status[name] = {
                    'trained': True,
                    'file_size_kb': round(stat.st_size / 1024, 1),
                    'trained_at': metadata.get('trained_at', 'unknown'),
                    'training_time_sec': metadata.get('training_time_sec', 0),
                    'metadata': metadata
                }
            else:
                status[name] = {
                    'trained': False,
                    'file_size_kb': 0,
                    'trained_at': None,
                    'training_time_sec': 0,
                    'metadata': {}
                }
        
        return status
    
    def clear_cache(self):
        """Clear all cached models (they remain on disk)."""
        self._models.clear()
        self._metadata.clear()


# Global registry instance
registry = ModelRegistry.get_instance()


def get_model(name: str) -> Optional[Any]:
    """Convenience function to get a model."""
    return registry.get_model(name)


def load_all_models() -> Dict[str, Any]:
    """Convenience function to load all models."""
    return registry.load_all_models()


def get_model_status() -> Dict[str, Dict]:
    """Convenience function to get all model statuses."""
    return registry.get_all_status()


def is_model_trained(name: str) -> bool:
    """Check if a specific model is trained."""
    return registry.is_trained(name)
