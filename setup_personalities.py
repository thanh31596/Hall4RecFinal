#!/usr/bin/env python3
"""
Setup script for personality generation
Helps users configure and run personality pre-generation
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import argparse

from utils import validate_api_setup, estimate_api_cost
from data_loader import MovieLensDataLoader

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'sentence_transformers', 
        'langchain', 'langchain_google_genai', 'tenacity'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_availability(data_path: str = "./ml-100k/"):
    """Check if MovieLens data is available"""
    data_path = Path(data_path)
    
    required_files = ["u.data", "u.user", "u.item", "u.genre", "u.occupation"]
    missing_files = []
    
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing MovieLens data files in {data_path}:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nDownload MovieLens 100K dataset:")
        print("   wget http://files.grouplens.org/datasets/movielens/ml-100k.zip")
        print("   unzip ml-100k.zip")
        return False
    
    print(f"✅ MovieLens data found in {data_path}")
    return True

def analyze_data_requirements(data_path: str = "./ml-100k/"):
    """Analyze data and estimate requirements"""
    try:
        data_loader = MovieLensDataLoader(data_path)
        data = data_loader.load_all_data()
        
        n_users = len(data['users'])
        n_items = len(data['items'])
        n_ratings = len(data['ratings'])
        
        print(f"\n📊 Dataset Analysis:")
        print(f"   Users: {n_users}")
        print(f"   Items: {n_items}")
        print(f"   Ratings: {n_ratings}")
        print(f"   Sparsity: {1 - (n_ratings / (n_users * n_items)):.1%}")
        
        # Estimate time and cost
        avg_prompt_length = 200  # Estimated
        avg_response_length = 100  # Estimated
        
        cost_info = estimate_api_cost(n_users, avg_prompt_length, avg_response_length)
        
        print(f"\n💰 Cost Estimation:")
        print(f"   Estimated prompts: {n_users}")
        print(f"   Estimated cost: ${cost_info['estimated_total_cost_usd']:.2f}")
        print(f"   Input tokens: ~{cost_info['estimated_input_tokens']:,.0f}")
        print(f"   Output tokens: ~{cost_info['estimated_output_tokens']:,.0f}")
        
        # Time estimation
        base_delay = 3.0  # seconds per request
        batch_delay = 15.0  # seconds between batches
        batch_size = 5
        
        total_batches = (n_users + batch_size - 1) // batch_size
        estimated_time = (n_users * base_delay) + (total_batches * batch_delay)
        
        print(f"\n⏱️  Time Estimation:")
        print(f"   Estimated time: {estimated_time / 60:.0f} minutes ({estimated_time / 3600:.1f} hours)")
        print(f"   Batch size: {batch_size}")
        print(f"   Total batches: {total_batches}")
        
        return {
            'n_users': n_users,
            'estimated_cost': cost_info['estimated_total_cost_usd'],
            'estimated_time_minutes': estimated_time / 60
        }
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return None

def create_generation_script(output_path: str = "./run_personality_generation.sh"):
    """Create a convenient shell script for personality generation"""
    script_content = f"""#!/bin/bash

# HallAgent4Rec Personality Generation Script
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 Starting HallAgent4Rec Personality Generation"
echo "================================================"

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY environment variable not set!"
    echo "Please set it with: export GOOGLE_API_KEY='your_api_key'"
    exit 1
fi

# Run personality generation with conservative settings
python generate_personalities.py \\
    --data_path ./ml-100k/ \\
    --output ./personalities.json \\
    --batch_size 5 \\
    --base_delay 3.0 \\
    --batch_delay 15.0

echo "✅ Personality generation completed!"
echo "You can now run the main analysis with:"
echo "python main.py --personalities_path ./personalities.json"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    print(f"✅ Created generation script: {output_path}")
    print(f"   Run with: ./{output_path}")

def interactive_setup():
    """Interactive setup process"""
    print("\n" + "="*60)
    print("🎯 HALLAGENT4REC PERSONALITY SETUP WIZARD")
    print("="*60)
    
    # Step 1: API Setup
    print("\n📋 Step 1: API Configuration")
    if not validate_api_setup():
        print("❌ Please configure your GOOGLE_API_KEY first:")
        print("   export GOOGLE_API_KEY='your_api_key_here'")
        return False
    
    # Step 2: Data Check
    print("\n📋 Step 2: Data Verification")
    data_path = input("Enter path to MovieLens 100K data [./ml-100k/]: ").strip() or "./ml-100k/"
    if not check_data_availability(data_path):
        return False
    
    # Step 3: Requirements Analysis
    print("\n📋 Step 3: Requirements Analysis")
    requirements = analyze_data_requirements(data_path)
    if not requirements:
        return False
    
    # Step 4: User Confirmation
    print(f"\n📋 Step 4: Confirmation")
    print(f"Ready to generate personalities for {requirements['n_users']} users")
    print(f"Estimated cost: ${requirements['estimated_cost']:.2f}")
    print(f"Estimated time: {requirements['estimated_time_minutes']:.0f} minutes")
    
    response = input("\nProceed with generation? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Setup cancelled by user")
        return False
    
    # Step 5: Configuration
    print(f"\n📋 Step 5: Configuration")
    output_path = input("Output file path [./personalities.json]: ").strip() or "./personalities.json"
    batch_size = input("Batch size [5]: ").strip() or "5"
    base_delay = input("Base delay between calls [3.0]: ").strip() or "3.0"
    
    try:
        batch_size = int(batch_size)
        base_delay = float(base_delay)
    except ValueError:
        print("❌ Invalid configuration values")
        return False
    
    # Step 6: Generate Command
    command = f"""python generate_personalities.py \\
    --data_path {data_path} \\
    --output {output_path} \\
    --batch_size {batch_size} \\
    --base_delay {base_delay}"""
    
    print(f"\n📋 Step 6: Ready to Generate")
    print("Run this command:")
    print(command)
    
    # Option to run now
    run_now = input("\nRun generation now? (y/n): ").strip().lower()
    if run_now == 'y':
        print("\n🚀 Starting personality generation...")
        os.system(command.replace('\\', ''))
    else:
        # Create script for later
        create_generation_script()
        print("\n✅ Setup completed! Run generation when ready.")
    
    return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup HallAgent4Rec Personality Generation')
    parser.add_argument('--interactive', action='store_true', help='Run interactive setup')
    parser.add_argument('--check-only', action='store_true', help='Only check requirements')
    parser.add_argument('--data_path', default='./ml-100k/', help='Path to MovieLens data')
    parser.add_argument('--create-script', action='store_true', help='Create generation script')
    
    args = parser.parse_args()
    
    print("🔧 HallAgent4Rec Personality Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data
    if not check_data_availability(args.data_path):
        sys.exit(1)
    
    if args.check_only:
        print("✅ All requirements satisfied")
        analyze_data_requirements(args.data_path)
        return
    
    if args.create_script:
        create_generation_script()
        return
    
    if args.interactive:
        if interactive_setup():
            print("\n🎉 Setup completed successfully!")
        else:
            print("\n❌ Setup failed or cancelled")
            sys.exit(1)
    else:
        # Quick check and analysis
        analyze_data_requirements(args.data_path)
        
        print("\n" + "="*50)
        print("NEXT STEPS")
        print("="*50)
        print("1. Set your API key:")
        print("   export GOOGLE_API_KEY='your_api_key'")
        print("\n2. Generate personalities:")
        print("   python generate_personalities.py")
        print("\n3. Run analysis:")
        print("   python main.py --personalities_path ./personalities.json")
        print("\nOr run interactive setup:")
        print("   python setup_personalities.py --interactive")

if __name__ == "__main__":
    main()
