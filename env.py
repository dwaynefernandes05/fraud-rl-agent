from typing import Tuple, Any, Dict
from models import Observation, Action, Reward
from tasks import generate_task_data

class FraudEnv:
    def __init__(self, task_level: str = "medium", num_users: int = 10):
        self.task_level = task_level
        self.dataset = generate_task_data(task_level, num_users)
        self.idx = 0
        self.total_users = len(self.dataset)

    def reset(self) -> Observation:
        self.idx = 0
        return self._get_obs()

    def state(self) -> Dict[str, Any]:
        """Returns the current state of the environment."""
        return {
            "task_level": self.task_level,
            "current_user_index": self.idx,
            "total_users": self.total_users,
            "is_done": self.idx >= self.total_users
        }

    def _get_obs(self) -> Observation:
        if self.idx >= self.total_users:
            return Observation(return_rate=0.0, account_age_days=0, linked_accounts=0)
        user = self.dataset[self.idx]
        return Observation(
            return_rate=user["return_rate"],
            account_age_days=user["account_age"],
            linked_accounts=user["linked_accounts"]
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        user = self.dataset[self.idx]
        is_fraud = user["is_fraud"]
        decision = action.decision.strip().upper()

        # Grader Logic: Must be between 0.0 and 1.0
        if decision == "REJECT":
            reward_val, info_text = (1.0, "Correctly rejected fraud") if is_fraud else (0.0, "False positive")
        elif decision == "APPROVE":
            reward_val, info_text = (0.0, "False negative") if is_fraud else (1.0, "Correctly approved legit")
        elif decision == "REVIEW":
            reward_val, info_text = (0.5, "Safe review, partial reward")
        else:
            reward_val, info_text = (0.0, "Invalid format penalty")

        reward = Reward(value=reward_val, reasoning=info_text)

        self.idx += 1
        done = self.idx >= self.total_users
        next_obs = self._get_obs()
        
        info = {"user_id": user["user_id"], "ground_truth": bool(is_fraud)}

        return next_obs, reward, done, info