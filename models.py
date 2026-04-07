from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    return_rate: float
    account_age_days: Optional[int] = None  # Now accepts null
    linked_accounts: Optional[int] = None   # Now accepts null

class Action(BaseModel):
    decision: str  

class Reward(BaseModel):
    value: float   
    reasoning: str