"""
Entry point untuk Coffee Grading System
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from grading_system import CoffeeGradingSystem


def main():
    """Main function"""
    print("=" * 50)
    print("SISTEM GRADING KOPI REALTIME")
    print("=" * 50)
    print()
    
    # Path ke model (jika sudah ada trained model)
    model_path = "models/coffee_grading_model.h5"
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model tidak ditemukan di {model_path}")
        print("Sistem akan menggunakan model baru (belum trained)")
        print("Untuk hasil terbaik, train model terlebih dahulu dengan data training")
        print()
        model_path = None
    
    # Initialize system
    try:
        system = CoffeeGradingSystem(model_path=model_path, camera_id=0)
        
        # Run realtime grading
        system.run_realtime()
        
    except KeyboardInterrupt:
        print("\nSistem dihentikan oleh user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
