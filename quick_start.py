#!/usr/bin/env python3
"""
HallAgent4Rec Quick Start Script
One-stop script to get HallAgent4Rec running with the new workflow
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

class QuickStart:
    """Quick start manager for HallAgent4Rec"""
    
    def __init__(self):
        self.current_dir = Path.cwd()
        
    def check_prerequisites(self) -> bool:
        """Check all prerequisites"""
        print("🔍 Checking prerequisites...")
        
        checks = [
            ("API Key", self._check_api_key),
            ("Dependencies", self._check_dependencies), 
            ("MovieLens Data", self._check_movielens_data),
            ("Required Scripts", self._check_required_scripts)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    print(f"  ✅ {check_name}")
                else:
                    print(f"  ❌ {check_name}")
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {check_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def _check_api_key(self) -> bool:
        """Check if Google API key is set"""
        api_key = os.getenv('GOOGLE_API_KEY')
        return api_key is not None and len(api_key) > 20
    
    def _check_dependencies(self) -> bool:
        """Check if required Python packages are installed"""
        required = [
            'numpy', 'pandas', 'scikit-learn', 'sentence_transformers',
            'langchain', 'langchain_google_genai', 'tenacity'
        ]
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                return False
        return True
    
    def _check_movielens_data(self) -> bool:
        """Check if MovieLens data is available"""
        data_path = self.current_dir / "ml-100k"
        required_files = ["u.data", "u.user", "u.item"]
        
        return data_path.exists() and all((data_path / f).exists() for f in required_files)
    
    def _check_required_scripts(self) -> bool:
        """Check if required scripts are present"""
        required_scripts = [
            "generate_personalities.py",
            "main.py", 
            "personality_generator.py",
            "data_loader.py"
        ]
        
        return all((self.current_dir / script).exists() for script in required_scripts)
    
    def run_full_pipeline(self, config: str = "small", skip_online: bool = False) -> bool:
        """Run the complete HallAgent4Rec pipeline"""
        print(f"\n🚀 Starting HallAgent4Rec Full Pipeline")
        print(f"Configuration: {config}")
        print(f"Skip online phase: {skip_online}")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Step 1: Generate personalities if not exists
            personalities_file = self.current_dir / "personalities.json"
            
            if not personalities_file.exists():
                print("\n📋 Step 1: Generating personality profiles...")
                print("⏱️  This will take 30-60 minutes...")
                
                cmd = [
                    sys.executable, "generate_personalities.py",
                    "--data_path", "./ml-100k/",
                    "--output", "./personalities.json",
                    "--batch_size", "5"
                ]
                
                result = subprocess.run(cmd, capture_output=False)
                if result.returncode != 0:
                    print("❌ Personality generation failed")
                    return False
                
                print("✅ Personality generation completed")
            else:
                print("\n📋 Step 1: Using existing personality profiles")
                
                # Validate existing personalities
                validation_cmd = [
                    sys.executable, "validate_personalities.py",
                    "--file", "./personalities.json",
                    "--summary"
                ]
                subprocess.run(validation_cmd, capture_output=False)
            
            # Step 2: Run main analysis
            print(f"\n📋 Step 2: Running main analysis...")
            
            main_cmd = [
                sys.executable, "main.py",
                "--personalities_path", "./personalities.json",
                "--config", config,
                "--experiment_name", f"quickstart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ]
            
            if skip_online:
                main_cmd.append("--skip_online")
            
            result = subprocess.run(main_cmd, capture_output=False)
            if result.returncode != 0:
                print("❌ Main analysis failed")
                return False
            
            print("✅ Main analysis completed")
            
            # Step 3: Show results
            print(f"\n📋 Step 3: Analysis summary")
            total_time = time.time() - start_time
            print(f"Total time: {total_time/60:.1f} minutes")
            
            # Find latest experiment
            experiments_dir = self.current_dir / "experiments"
            if experiments_dir.exists():
                latest_exp = max(experiments_dir.iterdir(), key=lambda x: x.stat().st_mtime)
                print(f"Results saved in: {latest_exp}")
                
                # Show final results if available
                results_file = latest_exp / "results" / "evaluation_main_evaluation.json"
                if results_file.exists():
                    import json
                    with open(results_file) as f:
                        results = json.load(f)
                    
                    if 'results' in results:
                        print("Key metrics:")
                        for metric, value in results['results'].items():
                            if 'HitRate' in metric or 'NDCG' in metric or 'RMSE' in metric:
                                print(f"  {metric}: {value:.4f}")
            
            print("\n🎉 Pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return False
    
    def run_quick_test(self) -> bool:
        """Run a quick test with minimal configuration"""
        print("\n🧪 Running quick test...")
        
        try:
            # Generate minimal personalities (if needed)
            personalities_file = self.current_dir / "personalities_test.json"
            
            if not personalities_file.exists():
                print("Generating test personalities (small subset)...")
                
                # Create a minimal test script
                test_script = f"""
import sys
sys.path.append('.')

from generate_personalities import PersonalityProfileGenerator
from data_loader import MovieLensDataLoader

# Load only first 10 users for testing
data_loader = MovieLensDataLoader('./ml-100k/')
data = data_loader.load_all_data()

# Limit to first 10 users
train_df, _ = data_loader.train_test_split(test_ratio=0.2)
train_matrix = data_loader.create_interaction_matrix(train_df)
user_demographics = data_loader.create_user_demographics()
item_metadata = data_loader.create_item_metadata()

# Restrict to first 10 users
train_matrix_small = train_matrix[:10, :]
user_demographics_small = {{k: v for k, v in user_demographics.items() if k < 10}}

# Generate personalities for subset
generator = PersonalityProfileGenerator('{personalities_file}')
generator.base_delay = 2.0  # Faster for testing
generator.batch_delay = 10.0

result = generator.generate_all_personalities('./ml-100k/', batch_size=3)
print("Test personality generation completed")
"""
                
                # Write and run test script
                test_script_file = self.current_dir / "test_generate.py"
                with open(test_script_file, 'w') as f:
                    f.write(test_script)
                
                result = subprocess.run([sys.executable, "test_generate.py"])
                test_script_file.unlink()  # Clean up
                
                if result.returncode != 0:
                    print("❌ Test personality generation failed")
                    return False
            
            # Run quick analysis
            print("Running quick analysis...")
            
            cmd = [
                sys.executable, "main.py",
                "--personalities_path", str(personalities_file),
                "--config", "tiny",
                "--skip_online",
                "--experiment_name", "quicktest"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Quick test passed!")
                return True
            else:
                print("❌ Quick test failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Quick test error: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Set up the environment for HallAgent4Rec"""
        print("🔧 Setting up HallAgent4Rec environment...")
        
        # Check if setup_personalities.py exists and run it
        setup_script = self.current_dir / "setup_personalities.py"
        if setup_script.exists():
            cmd = [sys.executable, str(setup_script), "--check-only"]
            result = subprocess.run(cmd, capture_output=False)
            return result.returncode == 0
        else:
            print("❌ setup_personalities.py not found")
            return False
    
    def show_help(self):
        """Show help and usage instructions"""
        help_text = """
🎯 HallAgent4Rec Quick Start Guide

Available commands:

1. Full Pipeline (Recommended):
   python quick_start.py --full
   
   - Generates personality profiles (30-60 min)
   - Runs complete analysis
   - Shows final results

2. Quick Test:
   python quick_start.py --test
   
   - Fast test with minimal data
   - Good for verifying setup

3. Setup Check:
   python quick_start.py --setup
   
   - Check prerequisites
   - Validate environment

4. Custom Configuration:
   python quick_start.py --full --config medium --skip-online
   
   - Use different model sizes: tiny, small, medium
   - Skip online LLM phase for faster execution

Prerequisites:
- Set GOOGLE_API_KEY environment variable
- Download MovieLens 100K dataset to ./ml-100k/
- Install required Python packages

For detailed setup instructions, run:
   python setup_personalities.py --interactive
"""
        print(help_text)

def main():
    """Main quick start function"""
    parser = argparse.ArgumentParser(description='HallAgent4Rec Quick Start')
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--setup', action='store_true', help='Setup and check environment')
    parser.add_argument('--config', default='small', choices=['tiny', 'small', 'medium'], help='Model configuration')
    parser.add_argument('--skip-online', action='store_true', help='Skip online LLM phase')
    parser.add_argument('--help-detailed', action='store_true', help='Show detailed help')
    
    args = parser.parse_args()
    
    starter = QuickStart()
    
    if args.help_detailed:
        starter.show_help()
        return
    
    # Default to showing help if no arguments
    if not any([args.full, args.test, args.setup]):
        starter.show_help()
        return
    
    print("🎯 HallAgent4Rec Quick Start")
    print("=" * 40)
    
    if args.setup:
        if starter.setup_environment():
            print("✅ Environment setup completed")
        else:
            print("❌ Environment setup failed")
            print("\nTry: python setup_personalities.py --interactive")
            sys.exit(1)
    
    elif args.test:
        print("Running quick test to verify setup...")
        
        if not starter.check_prerequisites():
            print("\n❌ Prerequisites not met")
            print("Run: python quick_start.py --setup")
            sys.exit(1)
        
        if starter.run_quick_test():
            print("\n✅ Quick test passed! Ready for full analysis.")
            print("Run: python quick_start.py --full")
        else:
            print("\n❌ Quick test failed. Check the error messages above.")
            sys.exit(1)
    
    elif args.full:
        print("Running full HallAgent4Rec pipeline...")
        
        if not starter.check_prerequisites():
            print("\n❌ Prerequisites not met")
            print("Run: python quick_start.py --setup")
            sys.exit(1)
        
        if starter.run_full_pipeline(args.config, args.skip_online):
            print("\n🎉 Full pipeline completed successfully!")
            print("Check the experiments/ directory for detailed results.")
            print("Run: streamlit run analysis_dashboard.py (to view results)")
        else:
            print("\n❌ Pipeline failed. Check the error messages above.")
            sys.exit(1)

if __name__ == "__main__":
    main()
