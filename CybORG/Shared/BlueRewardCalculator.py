from collections import namedtuple

from CybORG.Shared import Scenario
from CybORG.Shared.RedRewardCalculator import DistruptRewardCalculator, PwnRewardCalculator
from CybORG.Shared.RewardCalculator import RewardCalculator
import numpy as np


HostReward = namedtuple('HostReward','confidentiality availability')

class ConfidentialityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on compromise of hosts/data
    def __init__(self, agent_name: str, scenario: Scenario):
        self.scenario = scenario
        self.adversary = scenario.get_agent_info(agent_name).adversary
        super(ConfidentialityRewardCalculator, self).__init__(agent_name)
        self.infiltrate_rc = PwnRewardCalculator(self.adversary, scenario)
        self.compromised_hosts = {}
        self.subnet_confidentiality = {}
        self.host_to_subnet = {
            'User0': 'Subnet1',
            'User1': 'Subnet1',
            'User2': 'Subnet1',
            'User3': 'Subnet1',
            'User4': 'Subnet1',
            'Enterprise0': 'Subnet2',
            'Enterprise1': 'Subnet2',
            'Enterprise2': 'Subnet2',
            'Defender': 'Subnet2',
            'Op_Server0': 'Subnet3',
            'Op_Host0': 'Subnet3',
            'Op_Host1': 'Subnet3',
            'Op_Host2': 'Subnet3'
        }

    def reset(self):
        self.infiltrate_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.compromised_hosts = {}
        reward = -self.infiltrate_rc.calculate_reward(current_state, action, agent_observations, done)
        subnet_decomp = self._calculate_compromised_hosts()
        if reward != sum(subnet_decomp.values()):
            print("mismatch", reward, sum(subnet_decomp.values()))
        return reward, subnet_decomp


    def _calculate_compromised_hosts(self):
        self.subnet_confidentiality = {}  # New: subnet → cumulative value

        for host, value in self.infiltrate_rc.compromised_hosts.items():
            subnet = self.host_to_subnet.get(host, 'Unknown')
            if subnet not in self.subnet_confidentiality:
                self.subnet_confidentiality[subnet] = 0
            self.subnet_confidentiality[subnet] += -1 * value
            self.compromised_hosts[host] = -1 * value
        return self.subnet_confidentiality

class AvailabilityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on reduction in availability
    def __init__(self, agent_name: str, scenario: Scenario):
        super(AvailabilityRewardCalculator, self).__init__(agent_name)
        self.adversary = scenario.get_agent_info(agent_name).adversary
        self.disrupt_rc = DistruptRewardCalculator(self.adversary, scenario)
        self.impacted_hosts = {}

    def reset(self):
        self.disrupt_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.impacted_hosts = {}
        reward = -self.disrupt_rc.calculate_reward(current_state, action, agent_observations, done)
        self._calculate_impacted_hosts()
        #print(reward)
        return reward

    def _calculate_impacted_hosts(self):
        for host, value in self.disrupt_rc.impacted_hosts.items():
            self.impacted_hosts[host] = -1 * value

class HybridAvailabilityConfidentialityRewardCalculator(RewardCalculator):
    # Hybrid of availability and confidentiality reward calculator
    def __init__(self, agent_name: str, scenario: Scenario):
        super(HybridAvailabilityConfidentialityRewardCalculator, self).__init__(agent_name)
        self.availability_calculator = AvailabilityRewardCalculator(agent_name, scenario)
        self.confidentiality_calculator = ConfidentialityRewardCalculator(agent_name, scenario)

    def reset(self):
        self.availability_calculator.reset()
        self.confidentiality_calculator.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        availability = self.availability_calculator.calculate_reward(current_state, action, agent_observations, done)
        confidentiality, confidentiality_decomp = self.confidentiality_calculator.calculate_reward(current_state, action, agent_observations, done)
        reward = availability + confidentiality
        self._compute_host_scores(current_state.keys())
        return reward, availability, confidentiality, confidentiality_decomp

    def _compute_host_scores(self, hostnames):
        self.host_scores = {}
        compromised_hosts = self.confidentiality_calculator.compromised_hosts
        impacted_hosts = self.availability_calculator.impacted_hosts
        for host in hostnames:
            if host == 'success':
                continue
            compromised = compromised_hosts[host] if host in compromised_hosts else 0
            impacted = impacted_hosts[host] if host in impacted_hosts else 0
            reward_state = HostReward(compromised,impacted)  
                                    # confidentiality, availability
            self.host_scores[host] = reward_state

