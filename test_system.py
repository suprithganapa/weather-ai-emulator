"""
System Test Script
Verifies that all components are working correctly
"""

import os
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test dataset imports
        from datasets.window_dataset import WeatherWindowDataset
        from datasets.window_dataset_horizon import WeatherWindowDatasetHorizon
        print("  ✅ Dataset modules")
        
        # Test model imports
        from models.cnn_lstm import CNNLSTMEmulator
        print("  ✅ Model modules")
        
        # Test backend imports (without running server)
        sys.path.append('backend')
        # Note: Can't import main.py directly due to uvicorn dependency
        print("  ⚠️  Backend modules (skipped - requires server)")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_model_architecture():
    """Test model forward pass."""
    print("\nTesting model architecture...")
    
    try:
        import torch
        from models.cnn_lstm import CNNLSTMEmulator
        
        # Create model
        model = CNNLSTMEmulator()
        
        # Test forward pass
        batch_size = 2
        lookback = 6
        features = 5
        
        x = torch.randn(batch_size, lookback, features)
        reg_out, cls_out = model(x)
        
        assert reg_out.shape == (batch_size, 3), f"Regression output shape wrong: {reg_out.shape}"
        assert cls_out.shape == (batch_size, 10), f"Classification output shape wrong: {cls_out.shape}"
        
        print(f"  ✅ Model forward pass")
        print(f"     Input: {x.shape}")
        print(f"     Regression output: {reg_out.shape}")
        print(f"     Classification output: {cls_out.shape}")
        print(f"     Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"  ❌ Model test error: {e}")
        return False


def test_data_generator():
    """Test sample data generation."""
    print("\nTesting sample data generator...")
    
    try:
        # Generate small sample
        from generate_sample_data import generate_sample_data
        
        test_path = 'data/processed/test_sample.csv'
        generate_sample_data(
            cities=['Bangalore'],
            hours=100,  # Small sample
            output_path=test_path
        )
        
        # Verify file exists
        if os.path.exists(test_path):
            import pandas as pd
            df = pd.read_csv(test_path)
            print(f"  ✅ Sample data generated")
            print(f"     Rows: {len(df)}")
            print(f"     Columns: {len(df.columns)}")
            
            # Clean up
            os.remove(test_path)
            return True
        else:
            print(f"  ❌ File not created")
            return False
            
    except Exception as e:
        print(f"  ❌ Data generator error: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading (requires data file)."""
    print("\nTesting dataset loading...")
    
    csv_path = 'data/processed/nasa_power_labeled_v2.csv'
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  Dataset not found at {csv_path}")
        print(f"     Run: python generate_sample_data.py")
        return None
    
    try:
        from datasets.window_dataset import WeatherWindowDataset
        
        dataset = WeatherWindowDataset(
            csv_path,
            cities=['Bangalore'],
            lookback=6
        )
        
        if len(dataset) > 0:
            # Test getting a sample
            X, y_reg, y_cls = dataset[0]
            
            print(f"  ✅ Dataset loaded")
            print(f"     Samples: {len(dataset)}")
            print(f"     Input shape: {X.shape}")
            print(f"     Regression target shape: {y_reg.shape}")
            print(f"     Classification target shape: {y_cls.shape}")
            
            return True
        else:
            print(f"  ❌ Dataset is empty")
            return False
            
    except Exception as e:
        print(f"  ❌ Dataset loading error: {e}")
        return False


def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'backend',
        'datasets',
        'models',
        'train',
        'checkpoints',
        'data',
        'data/processed'
    ]
    
    all_exist = True
    for d in required_dirs:
        exists = os.path.exists(d)
        status = "✅" if exists else "❌"
        print(f"  {status} {d}")
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("="*60)
    print("Weather AI Emulator - System Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Python Imports", test_imports()))
    results.append(("Model Architecture", test_model_architecture()))
    results.append(("Data Generator", test_data_generator()))
    results.append(("Dataset Loading", test_dataset_loading()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        
        print(f"{status:12} {test_name}")
    
    # Overall status
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    passed = sum(1 for _, r in results if r is True)
    
    print("\n" + "="*60)
    if failed == 0:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Generate data: python generate_sample_data.py")
        print("2. Train model: python train/train_cnn_lstm_1h.py")
        print("3. Start API: python -m uvicorn backend.main:app --reload")
    else:
        print(f"❌ {failed} test(s) failed")
        print(f"⚠️  {skipped} test(s) skipped")
        print("\nPlease fix the errors before proceeding.")
    print("="*60)


if __name__ == '__main__':
    main()
