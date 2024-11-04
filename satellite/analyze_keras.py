import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_layer_info(layer):
    """Get layer information handling different layer types"""
    info = {
        "name": layer.name,
        "type": type(layer).__name__,
        "params": layer.count_params()
    }
    
    # Handle different layer types
    try:
        info["input_shape"] = layer.input_shape
    except AttributeError:
        try:
            info["input_shape"] = layer.input_spec.shape
        except AttributeError:
            info["input_shape"] = "Not available"
    
    try:
        info["output_shape"] = layer.output_shape
    except AttributeError:
        info["output_shape"] = "Not available"
    
    # For Bidirectional layers, get the wrapped layer info
    if isinstance(layer, tf.keras.layers.Bidirectional):
        info["forward_layer"] = layer.forward_layer.__class__.__name__
        info["backward_layer"] = layer.backward_layer.__class__.__name__
        info["merge_mode"] = layer.merge_mode
    
    return info

def analyze_model(model_path):
    """Analyze a single .keras model"""
    print(f"\nAnalyzing model: {model_path}")
    print("-" * 50)
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Print detailed layer information
    print("\nDetailed Layer Information:")
    for layer in model.layers:
        info = get_layer_info(layer)
        print(f"\nLayer: {info['name']}")
        print(f"Type: {info['type']}")
        print(f"Input shape: {info['input_shape']}")
        print(f"Output shape: {info['output_shape']}")
        print(f"Number of parameters: {info['params']}")
        
        # Print additional info for Bidirectional layers
        if isinstance(layer, tf.keras.layers.Bidirectional):
            print(f"Forward layer: {info['forward_layer']}")
            print(f"Backward layer: {info['backward_layer']}")
            print(f"Merge mode: {info['merge_mode']}")

def analyze_saved_models(models_dir='./saved_models'):
    """Analyze all .keras models in the saved_models directory"""
    # Get all .keras files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    
    if not model_files:
        print("No .keras files found in the saved_models directory")
        return
    
    print(f"Found {len(model_files)} model files:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    # Let user choose which model to analyze
    while True:
        try:
            choice = input("\nEnter the number of the model you want to analyze (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            choice = int(choice)
            if 1 <= choice <= len(model_files):
                model_path = os.path.join(models_dir, model_files[choice-1])
                analyze_model(model_path)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except Exception as e:
            print(f"Error analyzing model: {str(e)}")

if __name__ == "__main__":
    analyze_saved_models()
