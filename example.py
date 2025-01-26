import os
import wandb
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Load WandB environment variables
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")


if not all([wandb_api_key, wandb_project, wandb_entity]):
    print(
        "Please set WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY in the environment variables"
    )  # logged as DEBUG
    mode = "disabled"
else:
    print(f"Logging in with api key to project {wandb_project} for entity {wandb_entity}")
    wandb.login(key=wandb_api_key)
    mode = "online"

print(f"Mode: {mode}")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

run = wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    mode=mode,
    job_type="demo",
    name=f"example_{current_time}",
    notes="Example run",
    tags=["demo"],
    config={"learning_rate": 0.0, "architecture": "demo"},
)

# Log some metrics
for i in range(10):
    run.log({"metric": i})

run.finish()