---
title: Ecommerce Fraud Rl Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
- openenv
---
# E-Commerce Return Fraud Agent

## 🎯 Environment Description & Motivation
Rule-based fraud detection systems are rigid and easily bypassed by sophisticated actors, while traditional machine learning models lack the reasoning capabilities to handle missing data or nuanced edge cases. This OpenEnv environment simulates a real-world **E-Commerce Return Fraud Risk Assessment** desk. Agents must evaluate streaming transactional data and apply business logic to preserve Customer Lifetime Value (CLV) while aggressively halting return fraud.

This environment bridges the gap between static tabular data analysis and agentic reasoning, providing a benchmark for LLMs to demonstrate multi-step logical gating, boundary math, and null-value handling in a high-stakes financial context.

## 📊 Action and Observation Spaces

### Observation Space
The environment emits a JSON state representing a user's transaction profile:
* `return_rate` (float | null): The historical percentage of items returned.
* `account_age_days` (int | null): The tenure of the customer account.
* `linked_accounts` (int | null): Number of known alternate accounts matching the user's hardware/IP.

### Action Space
The agent must respond with a strictly formatted JSON object dictating the business action:
* `{"decision": "APPROVE"}`: Process the return normally.
* `{"decision": "REJECT"}`: Deny the return due to policy violation.
* `{"decision": "REVIEW"}`: Flag for human operator intervention (costs operational OPEX).

## 🏆 Task Descriptions & Difficulty

1. **Easy Task (Baseline Logic):** Tests basic rule adherence. Fraudulent behavior is obvious (e.g., extremely high return rates paired with brand-new accounts).
2. **Medium Task (Missing Data & Paranoia):** Introduces `null` values into the data stream. Agents must safely route ambiguous cases to `REVIEW` without hallucinating or panicking on missing constraints.
3. **Hard Task (Boundary Evasion):** Simulates real-world fraudsters intentionally aging their accounts or manipulating linked hardware profiles to sit exactly on the boundary of heuristic rules. Tests the model's mathematical precision and resistance to inherent bias against "perfect" `0.0` return rates.

## 🚀 Setup and Usage

**Local Installation:**
```bash
git clone [https://huggingface.co/spaces/YOUR_USERNAME/fraud-rl-agent](https://huggingface.co/spaces/YOUR_USERNAME/fraud-rl-agent)
cd fraud-rl-agent
pip install uv
uv lock
pip install openenv-core python-dotenv openai pydantic