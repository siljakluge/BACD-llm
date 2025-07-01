# These tests check that the green actions are working:
# DiscoverRemoteSystems, DiscoverNetworkServices, ExploitService, Escalate, Impact

# tests need to check that a range of inputs result in the correct changes to the state and return the correct obs
# tests should establish varying environmental states that results in these actions performing differently

from CybORG import CybORG
import inspect

from CybORG.Agents.SimpleAgents.GreenAgent import GreenAgent

def test_GreenAgent():
    # Create cyborg environment
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim',agents={'Green': GreenAgent})

    # Setup Agent
    action_space = cyborg.get_action_space('Green')
    session = action_space['session']
    assert session

