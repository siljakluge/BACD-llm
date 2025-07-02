import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random
import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from Agents.SHAPWrapper import SHAPWrapper

MAX_EPS = 100 # Adjust the number
number_steps = 10
agent_name = 'Blue'
random.seed(0)
os.environ['LLM_ENABLED'] = 'True'


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    commit_hash = "Not using git"
    # ask for a name
    name = "Johannes Loevenich"
    # ask for a team
    team = "Team.KI"
    # ask for a name for the agent
    name_of_agent = "PPO + Greedy decoys"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = MainAgent()
    agent = agent.load_meander()
    wrapper = SHAPWrapper(agent)

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in range(number_steps): #, 50, 100
        for red_agent in [RedMeanderAgent]: # B_lineAgent, RedMeanderAgent, SleepAgent
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            acts = []
            all_observations = []
            restore_costs = []
            availability_scores = []
            confidentiality_scores = []
            confidentiality_decomps = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action, observation_tmp = agent.get_action(observation, action_space)
                    acts.append(action)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    r.append(rew)
                    all_observations.append(observation_tmp)
                    # r.append(result.reward)
                    restore_costs.append(info["restore"])
                    availability_scores.append(info["availability"])
                    confidentiality_scores.append(info["confidentiality"])
                    confidentiality_decomps.append(info["confidentiality_decomp"])
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')


"""# Semantic Shit
semantic_labels = {
        0: "restore enterprise/opserver",
        1: "analyse enterprise/opserver",
        2: "remove enterprise/opserver",
        3: "analyse user hosts",
        4: "restore user hosts",
        5: "restore defender",
        6: "analyse defender",
        7: "remove defender/user hosts",
        8: "decoy"
    }

    # Your action groups (non-decoy)
action_groups = [
        [133, 134, 135, 139],  # 0
        [3, 4, 5, 9],  # 1
        [16, 17, 18, 22],  # 2
        [11, 12, 13, 14],  # 3
        [141, 142, 143, 144],  # 4
        [132],  # 5
        [2],  # 6
        [15, 24, 25, 26, 27]  # 7
    ]

    # Decoy actions
decoy_actions = {
        1000: [55, 107, 120, 29],
        1001: [43],
        1002: [44],
        1003: [37, 115, 76, 102],
        1004: [51, 116, 38, 90],
        1005: [130, 91],
        1006: [131],
        1007: [54, 106, 28, 119],
        1008: [61, 35, 113, 126]
    }
    # Map non-decoy actions
action_id_to_label = {}

for group_idx, group in enumerate(action_groups):
        label = semantic_labels[group_idx]
        for action_id in group:
            action_id_to_label[action_id] = label

    # Map decoy actions (all get the label "decoy")
for action_list in decoy_actions.values():
        for action_id in action_list:
            action_id_to_label[action_id] = semantic_labels[8]

semantic_actions = [action_id_to_label.get(a, f"UNKNOWN({a})") for a in acts]
    #print(semantic_actions)
"""
fixed_observations = []

for obs in all_observations:
    # If it's a torch tensor, convert and flatten
    if np.asarray(obs).shape != (62,):
        #print(np.asarray(obs).shape)
        obs = obs.squeeze()
        obs = obs.detach().cpu().numpy()

    # If still wrong shape (e.g. shape (1, 62)), flatten it
    obs = np.asarray(obs)
    if obs.ndim == 2 and obs.shape[0] == 1:
        obs = obs[0]  # from shape (1, 62) to (62,)

    # Now only accept if length == 62
    if obs.shape == (62,):
        fixed_observations.append(obs)
    else:
        print(f"[SKIPPED] Shape {obs.shape} — expected (62,)")

all_observations = np.array(fixed_observations)
unique_chunks = set()
count = 0
for obs in all_observations:
    for i in range(0, 52, 4):
        chunk = tuple(obs[i:i+4])
        unique_chunks.add(chunk)
        if chunk == (1.0, 1.0, 1.0, 1.0):
            count+=1

unique_chunks = sorted(list(unique_chunks))

for bits in unique_chunks:
    print(bits)
print(count)

sample = shap.sample(all_observations, 100)
explainer = shap.KernelExplainer(wrapper.predict, sample)

i = 42
obs = all_observations[42]
obs = obs.reshape(1, -1)

    # Compute SHAP values
shap_values = explainer.shap_values(sample)
shap_values_obs = explainer.shap_values(obs)
# print(shap_values.shape)

# Plot summary
#shap.summary_plot(list(shap_values[:,:,i] for i in range(36)), sample)

# ------- Group features -----------------------------------------------

# Define bit indices for each group
activity_bits = [i for i in range(0, 52, 4)] + [i for i in range(1, 52, 4)]
compromised_bits = [i for i in range(2, 52, 4)] + [i for i in range(3, 52, 4)]

# Optional: include internal state group (last 10 bits)
scan_state_bits = list(range(52, 62))

# Combine into final groups
groups = [activity_bits, compromised_bits, scan_state_bits]
group_names = ['Activity', 'Compromised', 'Scan State']

# Assuming shap_values shape: (samples, features, actions)
samples, _, num_actions = shap_values.shape

# Apply grouping
grouped_shap = np.stack([
    np.sum(shap_values[:, idxs, :], axis=1) for idxs in groups
], axis=1)  # shape: (samples, 3, actions)

# Dummy features for plotting
dummy_features = np.zeros((samples, len(groups)))


def decode_host_bits(bits):
    activity_bits = tuple(bits[:2])
    compromised_bits = tuple(bits[2:4])

    activity_map = {
        (0, 0): 'None',
        (1, 0): 'Scan',
        (1, 1): 'Exploit'
    }
    compromised_map = {
        (0, 0): 'No',
        (1, 0): 'Unknown',
        (0, 1): 'User',
        (1, 1): 'Privileged'
    }

    activity = activity_map.get(activity_bits, 'Invalid')
    compromised = compromised_map.get(compromised_bits, 'Invalid')

    return activity, compromised

def decode_all_hosts(obs):
    return [decode_host_bits(obs[i:i+4]) for i in range(0, 52, 4)]

# Initialize structures
categories = [
    'Activity: None', 'Activity: Scan', 'Activity: Exploit',
    'Compromised: No', 'Compromised: Unknown',
    'Compromised: User', 'Compromised: Privileged'
]

samples, _, num_actions = shap_values.shape
grouped_shap_decomp = np.zeros((samples, len(categories), num_actions))

# For each sample
for sample_idx in range(samples):
    obs = all_observations[sample_idx]
    decoded_hosts = decode_all_hosts(obs)

    for host_idx, (activity, compromised) in enumerate(decoded_hosts):
        base = host_idx * 4

        # Extract the SHAP values for this host (4 bits)
        host_shap = shap_values[sample_idx, base:base+4, :]

        # Add SHAP contributions to appropriate groups
        if activity in ['None', 'Scan', 'Exploit']:
            category_idx = categories.index(f'Activity: {activity}')
            grouped_shap_decomp[sample_idx, category_idx, :] += host_shap[0:2, :].sum(axis=0)

        if compromised in ['No', 'Unknown', 'User', 'Privileged']:
            category_idx = categories.index(f'Compromised: {compromised}')
            grouped_shap_decomp[sample_idx, category_idx, :] += host_shap[2:4, :].sum(axis=0)

dummy_sample = np.zeros((samples, len(categories)))


host_states = ["noactivity_notcompromised", "noactivity_useraccess", "noactivity_priviledgedaccess", "scanned_notcompromised", "exploit_useraccess", "exploit_priviledgedaccess"]
grouped_shap_hosts = np.zeros((samples, len(host_states), num_actions))

for sample_idx in range(samples):
    obs = all_observations[sample_idx]

    for host_idx in range(13):
        base = host_idx * 4

        host_shap = shap_values[sample_idx, base:base+4, :]
        no_no = np.array([0.0, 0.0, 0.0, 0.0])
        no_user = np.array([0.0, 0.0, 0.0, 1.0])
        no_priviledged = np.array([0.0, 0.0, 1.0, 1.0])
        scan_no = np.array([1.0, 0.0, 0.0, 0.0])
        exploit_user = np.array([1.0, 1.0, 0.0, 1.0])
        exploit_priviledged = np.array([1.0, 1.0, 1.0, 1.0])

        if np.array_equal(np.array(obs[base:base+4]), no_no):
            grouped_shap_hosts[sample_idx, 0,:] += host_shap[0:4,:].sum(axis=0)
        elif np.array_equal(np.array(obs[base:base+4]), no_user):
            grouped_shap_hosts[sample_idx, 1, :] += host_shap[0:4, :].sum(axis=0)
        elif np.array_equal(np.array(obs[base:base+4]), no_priviledged):
            grouped_shap_hosts[sample_idx, 2, :] += host_shap[0:4, :].sum(axis=0)
        elif np.array_equal(np.array(obs[base:base + 4]), scan_no):
            grouped_shap_hosts[sample_idx, 3, :] += host_shap[0:4, :].sum(axis=0)
        elif np.array_equal(np.array(obs[base:base + 4]), exploit_user):
            grouped_shap_hosts[sample_idx, 4, :] += host_shap[0:4, :].sum(axis=0)
        elif np.array_equal(np.array(obs[base:base + 4]), exploit_priviledged):
            grouped_shap_hosts[sample_idx, 5, :] += host_shap[0:4, :].sum(axis=0)

dummy = np.zeros((samples, len(host_states)))

#Group definitions
action_groups = [
    [133, 134, 135, 139],  # restore enterprise/opserver
    [3, 4, 5, 9],  # analyse enterprise/opserver
    [16, 17, 18, 22],  # remove enterprise/opserver
    [11, 12, 13, 14],  # analyse user hosts
    [141, 142, 143, 144],  # restore user hosts
    [132],  # restore defender
    [2],  # analyse defender
    [15, 24, 25, 26, 27],  # remove defender/user hosts
    list(range(36 - 9, 36))  # last 9 actions = decoys
]
# Action group labels
action_labels = [
    "restore enterprise/opserver",
    "analyse enterprise/opserver",
    "remove enterprise/opserver",
    "analyse user hosts",
    "restore user hosts",
    "restore defender",
    "analyse defender",
    "remove defender/user hosts",
    "decoys"
]

action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                         132, 2, 15, 24, 25, 26, 27]

# Map from action ID to index in shap_values
action_id_to_index = {aid: idx for idx, aid in enumerate(action_space)}

# Use that to convert your logical groupings
action_groups_idx = [
    [action_id_to_index[aid] for aid in group if aid in action_id_to_index]
    for group in action_groups
]
samples, num_features, _ = grouped_shap.shape
grouped_shap_actions = np.zeros((samples, num_features, len(action_groups_idx)))
grouped_shap_actions_decomp = np.zeros((samples, len(categories), len(action_groups_idx)))
grouped_shap_actions_hosts = np.zeros((samples, len(host_states), len(action_groups_idx)))

for a_idx, group in enumerate(action_groups_idx):
    for idx in group:
        grouped_shap_actions[:, :, a_idx] += grouped_shap[:, :, idx]
        grouped_shap_actions_decomp[:, :, a_idx] += grouped_shap_decomp[:, :, idx]
        grouped_shap_actions_hosts[:, :, a_idx] += grouped_shap_hosts[:, :, idx]
#print(grouped_shap_actions.shape)

# + Grouped actions
#shap.summary_plot(list(grouped_shap_actions[:, :, i] for i in range(9)), dummy_features, feature_names=group_names)

regrouped_action_groups = [
    [1, 3, 6],  # Analyse: enterprise/opserver, user hosts, defender
    [0, 4, 5],  # Restore: enterprise/opserver, user hosts, defender
    [2, 7],  # Remove: enterprise/opserver, defender/user hosts
    [8]  # Decoys
]

regrouped_labels = [
    "Analyse",
    "Restore",
    "Remove",
    "Decoys"
]
final_shap = np.zeros((samples, num_features, len(regrouped_action_groups)))
final_shap_decomp = np.zeros((samples, len(categories), len(regrouped_action_groups)))
final_shap_hosts = np.zeros((samples, len(host_states), len(regrouped_action_groups)))

for g_idx, group in enumerate(regrouped_action_groups):
    for idx in group:
        final_shap[:, :, g_idx] += grouped_shap_actions[:, :, idx]
        final_shap_decomp[:, :, g_idx] += grouped_shap_actions_decomp[:, :, idx]
        final_shap_hosts[:, :, g_idx] += grouped_shap_actions_hosts[:, :, idx]

shap.summary_plot(
    list(final_shap[:, :, i] for i in range(len(regrouped_labels))),
    dummy_features,
    feature_names = group_names,
    plot_type = "bar",
    class_names = regrouped_labels
)
shap.summary_plot(
    list(final_shap_decomp[:, :, i] for i in range(len(regrouped_labels))),
    dummy_sample,
    feature_names=categories,
    plot_type="bar",
    class_names=regrouped_labels
)
shap.summary_plot(
    list(final_shap_hosts[:, :, i] for i in range(len(regrouped_labels))),
    dummy,
    feature_names=host_states,
    plot_type="bar",
    class_names=regrouped_labels
)
# SHAP with features as subnets

host_list = [
    'Defender',
    'Enterprise0',
    'Enterprise1',
    'Enterprise2',
    'Op_Host0',
    'Op_Host1',
    'Op_Host2',
    'Op_Server0',
    'User0',
    'User1',
    'User2',
    'User3',
    'User4'
]

host_to_subnet = {
    'User0': 'Subnet1', 'User1': 'Subnet1', 'User2': 'Subnet1', 'User3': 'Subnet1', 'User4': 'Subnet1',
    'Enterprise0': 'Subnet2', 'Enterprise1': 'Subnet2', 'Enterprise2': 'Subnet2', 'Defender': 'Subnet2',
    'Op_Server0': 'Subnet3', 'Op_Host0': 'Subnet3', 'Op_Host1': 'Subnet3', 'Op_Host2': 'Subnet3'
}

subnet_features = {
    'Subnet1': [],
    'Subnet2': [],
    'Subnet3': []
}

for i, host in enumerate(host_list):
    subnet = host_to_subnet[host]
    # feature indices for host i
    features_for_host = [4*i + j for j in range(4)]
    subnet_features[subnet].extend(features_for_host)

# Add scan_state features (52 to 61)
subnet_features['scan_state'] = list(range(52, 62))

#print(subnet_features)
samples, _, actions = shap_values.shape
grouped_shap_by_subnet = np.zeros((samples, len(subnet_features), actions))

subnet_names = list(subnet_features.keys())

for i, subnet in enumerate(subnet_names):
    idxs = subnet_features[subnet]
    grouped_shap_by_subnet[:, i, :] = np.sum(shap_values[:, idxs, :], axis=1)

samples, num_subnets, _ = grouped_shap_by_subnet.shape
grouped_shap_actions = np.zeros((samples, num_subnets, len(action_groups)))

for a_idx, group in enumerate(action_groups):
    for aid in group:
        if aid in action_id_to_index:
            idx = action_id_to_index[aid]
            grouped_shap_actions[:, :, a_idx] += grouped_shap_by_subnet[:, :, idx]

final_shap = np.zeros((samples, num_subnets, len(regrouped_action_groups)))

for g_idx, group in enumerate(regrouped_action_groups):
    for idx in group:
        final_shap[:, :, g_idx] += grouped_shap_actions[:, :, idx]

dummy_sample = np.zeros((samples, num_subnets))

shap.summary_plot(
    [final_shap[:, :, i] for i in range(len(regrouped_labels))],
    dummy_sample,
    feature_names=list(subnet_features.keys()),
    plot_type="bar",
    class_names=regrouped_labels
)

#--------------------------------------------------------------------------------------------------------------------
# REWARD DECOMPOSITION
number_episodes = 10

plt.plot(availability_scores, label="Availability Penalty")
plt.plot(confidentiality_scores, label="Confidentiality Penalty")
plt.plot(restore_costs, label="Restore Penalty")

plt.title("Reward Components Over Time")
plt.xlabel("Step")
plt.xlim(0, number_steps * number_episodes)
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.show()

dict_list = confidentiality_decomps
for i, d in enumerate(dict_list):
    d["restore"] = restore_costs[i]
# Step 1: find all keys used in any dict
all_keys = set()
for d in dict_list:
    all_keys.update(d.keys())

# Step 2: update each dict to have all keys, missing keys get value 0
for d in dict_list:
    for key in all_keys:
        if key not in d:
            d[key] = 0

#print(dict_list)

# Extract all keys
all_keys = list(dict_list[0].keys())

# Prepare data arrays
data = {key: np.array([d[key] for d in dict_list]) for key in all_keys}
x = np.arange(len(dict_list))

colors = plt.cm.tab10.colors

# Separate stacking for positive and negative values
pos_bottom = np.zeros(len(dict_list))
neg_bottom = np.zeros(len(dict_list))

for i, key in enumerate(all_keys):
    values = data[key]

    pos_values = np.where(values > 0, values, 0)
    neg_values = np.where(values < 0, values, 0)

    # Stack negative bars
    plt.bar(x, neg_values, width=1.2, bottom=neg_bottom, color=colors[i % len(colors)], label=key)
    neg_bottom += neg_values

plt.xlabel('Index')
plt.xlim(0, number_steps * number_episodes)
plt.ylim(-2,0)
plt.ylabel('Reward value')
plt.title('Stacked rewards per subnet')
plt.legend()
plt.show()
