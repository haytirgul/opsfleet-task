"""
Verify that the project setup is correct.

This script checks:
1. Environment variables are loaded
2. Google API key is set
3. Required directories exist
4. Required packages are installed
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check_env_file():
    """Check if .env file exists."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        print("[OK] .env file found")
        return True
    else:
        print("[FAIL] .env file NOT found")
        print("  Please create a .env file based on .env.example")
        return False

def check_api_key():
    """Check if Google API key is set."""
    try:
        from settings import GOOGLE_API_KEY
        if GOOGLE_API_KEY:
            print(f"[OK] Google API key loaded (starts with: {GOOGLE_API_KEY[:20]}...)")
            return True
        else:
            print("[FAIL] Google API key is NOT set")
            print("  Please add GOOGLE_API_KEY to your .env file")
            return False
    except Exception as e:
        print(f"[FAIL] Error loading settings: {e}")
        return False

def check_directories():
    """Check if required directories exist."""
    from settings import DATA_DIR, INPUT_DIR, OUTPUT_DIR
    
    all_exist = True
    for dir_path, name in [(DATA_DIR, "data"), (INPUT_DIR, "input"), (OUTPUT_DIR, "output")]:
        if dir_path.exists():
            print(f"[OK] {name} directory exists: {dir_path}")
        else:
            print(f"[FAIL] {name} directory NOT found: {dir_path}")
            all_exist = False
    
    return all_exist

def check_input_data():
    """Check if the input data file exists."""
    from settings import INPUT_DIR
    
    input_file = INPUT_DIR / "langgraph_llms_full.txt"
    if input_file.exists():
        size_mb = input_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Input data file found: {input_file.name} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"[FAIL] Input data file NOT found: {input_file}")
        print("  Please download from: https://langchain-ai.github.io/langgraph/llms-full.txt")
        return False

def check_packages():
    """Check if required packages are installed."""
    required_packages = [
        "langchain",
        "langgraph",
        "langchain_google_genai",
        "langchain_chroma",
        "chromadb",
        "dotenv",
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package} installed")
        except ImportError:
            print(f"[FAIL] {package} NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\n  Install missing packages with: pip install -r requirements.txt")
    
    return all_installed

def test_api_connection():
    """Test Google Gemini API connection."""
    try:
        from settings import GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        print("\nTesting Google Gemini API connection...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        
        response = llm.invoke("Say 'API test successful' in exactly those words.")
        print(f"[OK] API connection successful!")
        print(f"  Response: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"[FAIL] API connection failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("="*70)
    print("PROJECT SETUP VERIFICATION")
    print("="*70 + "\n")
    
    checks = [
        ("Environment File", check_env_file),
        ("API Key", check_api_key),
        ("Directories", check_directories),
        ("Input Data", check_input_data),
        ("Packages", check_packages),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        results.append(check_func())
    
    # Optional API test
    print(f"\nAPI Connection Test (optional):")
    print("-" * 40)
    api_result = test_api_connection()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"\n[OK] All checks passed! ({passed}/{total})")
        if api_result:
            print("[OK] API connection test also passed!")
        print("\nYou're ready to run the ingestion:")
        print("  python scripts/ingest.py")
    else:
        print(f"\n[FAIL] {total - passed} check(s) failed. Please fix the issues above.")
        print("\nRefer to the setup instructions in README.md")
    
    print()

if __name__ == "__main__":
    main()

