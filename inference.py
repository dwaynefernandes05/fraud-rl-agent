import os
import json
import time
from openai import OpenAI
from env import FraudEnv
from dotenv import load_dotenv  
from models import Action

# Strictly required OpenEnv Variables
load_dotenv()
api_key = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY"))
base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini") # Swap to your preferred default

if not api_key:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set.")

client = OpenAI(api_key=api_key, base_url=base_url)

def run_evaluation(task_level: str, num_users: int = 5):
    env = FraudEnv(task_level=task_level, num_users=num_users)
    obs = env.reset()
    
    print(f"[START] Task: {task_level} fraud detection")
    
    while True:
        state = env.state()
        if state["is_done"]:
            break
            
        print(f"[STEP] Obs: {obs.model_dump()}")
        
        system_prompt = """You are an elite fraud detection agent. 
Respond ONLY with a JSON object containing a single key "decision" with value "APPROVE", "REJECT", or "REVIEW".

EVALUATE THESE RULES IN EXACT ORDER:
Rule 1: If linked_accounts is strictly greater than 3, you MUST output "REJECT".
Rule 2: If return_rate > 0.70 AND account_age_days < 60, you MUST output "REJECT".
Rule 3: If any value is null AND return_rate > 0.60, you MUST output "REVIEW".
Rule 4: If NONE of the above rules are broken, you MUST output "APPROVE".

CRITICAL EXAMPLES FOR ALIGNMENT:
- Obs: {"return_rate": 0.0, "account_age_days": 270, "linked_accounts": 1}
  Decision: "APPROVE" (Breaks no rules. Must use Rule 4).
- Obs: {"return_rate": 0.08, "account_age_days": null, "linked_accounts": null}
  Decision: "APPROVE" (Has nulls, but return_rate is NOT > 0.60. Must use Rule 4).
- Obs: {"return_rate": 0.95, "account_age_days": 200, "linked_accounts": 2}
  Decision: "APPROVE" (Age is > 60, so Rule 2 fails. Must use Rule 4)."""

        try:
            response = client.chat.completions.create(
                model=model_name,
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(obs.model_dump())}
                ],
                temperature=0.0
            )
            parsed = json.loads(response.choices[0].message.content)
            decision = parsed.get("decision", "REVIEW").upper()
            
        except Exception as e:
            # THIS WILL PRINT YOUR ACTUAL ERROR INSTEAD OF SILENTLY FAILING
            print(f"[API ERROR]: {e}") 
            decision = "REVIEW"

        action = Action(decision=decision)
        next_obs, reward, done, info = env.step(action)
        
        print(f"[STEP] Action: {action.model_dump()} | Reward: {reward.model_dump()}")
        
        obs = next_obs

        time.sleep(2)
        
    print(f"[END] Task: {task_level} Complete\n")

if __name__ == "__main__":
    run_evaluation("easy")
    run_evaluation("medium")
    run_evaluation("hard")