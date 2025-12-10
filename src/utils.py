import os
import yaml
import wandb
from dotenv import load_dotenv

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file and env variables.
    """
    # 1. Load environment variables from .env file (if present)
    load_dotenv()
    
    # 2. Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # 3. Parse YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def init_wandb(config):
    """
    Initializes Weights & Biases using the project name from config
    and the API key from the environment.
    """
    # Check if API key is loaded
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("⚠️  WARNING: WANDB_API_KEY not found in .env or environment variables.")
        print("   W&B logging might fail or require manual login.")
    else:
        # Explicit login ensures we use the key from .env
        wandb.login(key=api_key)

    # Initialize the run
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
        reinit=True
    )
    print(f"✅ W&B Initialized: {config['project_name']} / {config['run_name']}")