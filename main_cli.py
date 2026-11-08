import os

from dotenv import load_dotenv
from huggingface_hub import login

from src.rag.cli import main
from src.rag.config import PROJECT_ROOT

if __name__ == "__main__":
    # Load .env file
    load_dotenv(PROJECT_ROOT / ".env")
    # Try setting HF credentials.
    # Not required for public datasets.
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        print("HF_TOKEN found; authenticating.")
        login(token=HF_TOKEN)
    main()
