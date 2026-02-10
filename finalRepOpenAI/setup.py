#!/usr/bin/env python3
"""
WMAC 2026 Research Setup Script
Sets up the complete research environment for emergent communication protocols
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "openai>=1.0.0",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/simulations",
        "data/analysis",
        "logs",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")

def verify_structure():
    """Verify repository structure"""
    print("🔍 Verifying repository structure...")
    
    required_files = [
        "game_environment/advanced_collusion_agent.py",
        "game_environment/mixed_player_communication_game.py",
        "wmac2026/run_wmac.py",
        "analysis/analysis_pipeline.py",
        "team_coordination_engine.py",
        "llm_prompts.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files present")
    return True

def create_test_script():
    """Create a test script to verify setup"""
    test_script = """#!/usr/bin/env python3
\"\"\"
Test script to verify WMAC 2026 research setup
\"\"\"

import sys
import os
from pathlib import Path

def test_imports():
    \"\"\"Test critical imports\"\"\"
    try:
        from game_environment.advanced_collusion_agent import AdvancedCollusionAgent
        from game_environment.mixed_player_communication_game import MixedPlayerCommunicationGame
        from team_coordination_engine import TeamCoordinationEngine
        from wmac2026.run_wmac import main
        print("✅ All critical imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_directory_structure():
    \"\"\"Test directory structure\"\"\"
    required_dirs = ["game_environment", "wmac2026", "analysis", "texasholdem"]
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Directory structure correct")
    return True

def main():
    \"\"\"Run all tests\"\"\"
    print("🧪 Testing WMAC 2026 research setup...")
    
    tests = [
        test_imports,
        test_directory_structure
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Setup verification successful!")
        print("\\n🚀 Ready to run research simulations:")
        print("   python3 wmac2026/run_wmac.py --hands 20 --collusion-llm-players 2 --coordination-mode emergent_only --output-dir data/test_simulation")
    else:
        print("❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    # Make it executable
    os.chmod("test_setup.py", 0o755)
    print("✅ Created test_setup.py")

def main():
    """Main setup function"""
    print("🎯 WMAC 2026 Research Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Verify structure
    if not verify_structure():
        print("❌ Repository structure verification failed")
        sys.exit(1)
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Run setup verification: python3 test_setup.py")
    print("2. Start research simulations: python3 wmac2026/run_wmac.py --help")
    print("3. Run analysis pipeline: python3 analysis/run_complete_analysis.py")
    print("\n📚 Documentation: README.md")
    print("🔬 Research Methodology: WMACH_2026_RESEARCH_METHODOLOGY.md")

if __name__ == "__main__":
    main()