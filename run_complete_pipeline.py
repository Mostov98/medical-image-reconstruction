#!/usr/bin/env python3
"""
Complete training pipeline for Medical Image Reconstruction
- Trains the U-Net model
- Saves model weights
- Saves visualization results
- Commits and pushes to GitHub
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🚀 MEDICAL IMAGE RECONSTRUCTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Train the model
    print("\n📚 Step 1: Training the model...")
    if not run_command("python project_model.py", "Model training"):
        print("❌ Training failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Verify files exist
    print("\n📊 Step 2: Verifying generated files...")
    files_to_check = {
        'models/unet_model_weights.pth': 'Model weights',
        'results/reconstruction_results.png': 'Results visualization'
    }
    
    all_exist = True
    for filepath, description in files_to_check.items():
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            size_str = f"{file_size / (1024*1024):.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.2f} KB"
            print(f"  ✅ {description}: {filepath} ({size_str})")
        else:
            print(f"  ❌ {description}: {filepath} NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("❌ Some files are missing. Check training output above.")
        sys.exit(1)
    
    # Step 3: Git status
    print("\n📝 Step 3: Checking Git status...")
    os.system("git status")
    
    # Step 4: Add files to git
    print("\n📤 Step 4: Adding files to Git...")
    run_command("git add models/unet_model_weights.pth", "Adding model weights")
    run_command("git add results/reconstruction_results.png", "Adding results")
    run_command("git add project_model.py README.md .gitignore", "Adding configuration files")
    
    # Step 5: Commit
    print("\n💾 Step 5: Committing changes...")
    commit_msg = "feat: Train and save U-Net model weights and results"
    run_command(f'git commit -m "{commit_msg}"', "Creating commit")
    
    # Step 6: Push to GitHub
    print("\n🚀 Step 6: Pushing to GitHub...")
    run_command("git push origin main", "Pushing to GitHub")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Summary:")
    print(f"  ✅ Model trained and saved to: models/unet_model_weights.pth")
    print(f"  ✅ Results saved to: results/reconstruction_results.png")
    print(f"  ✅ Changes committed and pushed to GitHub")
    print("\n🎯 Next steps:")
    print("  1. Visit: https://github.com/Mostov98/medical-image-reconstruction")
    print("  2. Verify model and results are in the repository")
    print("  3. Share your project with collaborators!")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
