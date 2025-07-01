# Agents/SHAPWrapper.py

import numpy as np
import torch

class SHAPWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.full_observation_template = np.zeros(62, dtype=int)

    def predict(self, X):
        #outputs = []
        #for observations in X:
        state = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        #    action_probs = self.agent.get_action_probabilities(state)
        #    outputs.append(action_probs)
        action_probs = self.agent.get_action_probabilities(state)
        action_probs = np.array(np.squeeze(action_probs, axis=0))
        return action_probs
