# src/test_model.py
"""
Unit tests for ML model
"""
import unittest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model

class TestModel(unittest.TestCase):
    
    def test_model_training(self):
        """Test that model trains successfully"""
        accuracy = train_model()
        self.assertIsNotNone(accuracy)
        self.assertGreater(accuracy, 0.0)
    
    def test_model_accuracy_threshold(self):
        """Test that model achieves minimum accuracy"""
        accuracy = train_model()
        self.assertGreater(accuracy, 0.85, "Model accuracy below threshold")
    
    def test_model_saved(self):
        """Test that model file is created"""
        train_model()
        self.assertTrue(os.path.exists('models/model.pkl'))

if __name__ == '__main__':
    unittest.main()