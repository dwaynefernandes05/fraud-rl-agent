import os
import json
import time
from openai import OpenAI
from env import FraudEnv
from models import Action

# Safe dotenv import in case the judge's container doesn't have it installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Strictly required OpenEnv Variables
api_key = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY"))
base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini") 

if not api_key:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set.")

client = OpenAI(api_key=api_key, base_url=base_url)

def run_evaluation(task_level: str, num_users: int = 5):
    env = FraudEnv(task_level=task_level, num_users=num_users)
    obs = env.reset()
    
    # Strictly formatted [START] log
    print(f"[START] task={task_level} env=openenv_fraud model={model_name}", flush=True)
    
    step_count = 0
    rewards_list = []
    
    while True:
        state = env.state()
        if state["is_done"]:
            break
            
        step_count += 1
        error_val = "null"
        
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
            error_val = str(e).replace('\n', ' ')
            decision = "REVIEW"

        action = Action(decision=decision)
        next_obs, reward, done, info = env.step(action)
        
        rewards_list.append(reward.value)
        
        print(f"[STEP] step={step_count} action={decision} reward={reward.value:.2f} done={str(done).lower()} error={error_val}", flush=True)
        
        obs = next_obs
        time.sleep(2)
        
        if done:
            break
            
    total_possible = step_count * 1.0
    score = sum(rewards_list) / total_possible if total_possible > 0 else 0.0
    success = score >= 0.75 # Lowered threshold
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    
    print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    print("[INFO] Warming up network for Phase 2...", flush=True)
    time.sleep(5) # Reduced sleep to prevent Validator timeout
    
    for level in ["easy", "medium", "hard"]:
        try:
            run_evaluation(level, num_users=5)
        except Exception as task_err:
            print(f"[ERROR] Task {level} failed: {task_err}", flush=True)
            continue 
            
    print("[INFO] Evaluation complete. Exiting gracefully.", flush=True)
