# Copyright DST Group. Licensed under the MIT license.

import pprint
from copy import deepcopy

from CybORG.Shared.Observation import Observation
from CybORG.Tests.test_sim.test_Acceptance.test_reward_function import availability_values


class Results:

    def __init__(self,
                 observation: dict = None,
                 done: bool = None,
                 reward: float = None,
                 info=None,
                 parameter_mask=None,
                 action_space=None,
                 error: Exception = None,
                 error_msg: str = None,
                 next_observation=None,
                 action=None,
                 action_name: str = None,
                 restore: float = None,
                 availability: float = None,
                 confidentiality: float = None,
                 confidentiality_decomp: dict = None):
        self.observation = observation
        self.next_observation = next_observation
        self.done = done
        self.reward = reward
        self.action = action
        self.info = info
        self.parameter_mask = parameter_mask
        self.action_space = action_space
        self.error = error
        self.error_msg = error_msg
        self.action_name = action_name
        self.restore = restore
        self.availability = availability
        self.confidentiality = confidentiality
        self.confidentiality_decomp = confidentiality_decomp
        self.selection_masks = None

    def has_error(self):
        return self.error is not None

    def copy(self):
        copy_kwargs = {
            "done": self.done,
            "reward": self.reward,
            "error": deepcopy(self.error),
            "error_msg": deepcopy(self.error_msg),
            "action": deepcopy(self.action),
            "info": deepcopy(self.info),
            "action_space": deepcopy(self.action_space)
        }

        if isinstance(self.observation, Observation):
            copy_kwargs["observation"] = self.observation.copy()
        else:
            copy_kwargs["observation"] = deepcopy(self.observation)

        if isinstance(self.next_observation, Observation):
            copy_kwargs["next_observation"] = self.next_observation.copy()
        else:
            copy_kwargs["next_observation"] = deepcopy(self.next_observation)

        return Results(**copy_kwargs)

    def __str__(self):
        output = [f"{self.__class__.__name__}:"]
        for attr, v in self.__dict__.items():
            if v is None:
                continue
            if isinstance(v, dict):
                v_str = pprint.pformat(v)
            else:
                v_str = str(v)
            output.append(f"{attr}={v_str}")
        return "\n".join(output)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        for k, v in self.__dict__.items():
            if k not in other.__dict__:
                return False
            if v != other.__dict__[k]:
                return False
        return True
