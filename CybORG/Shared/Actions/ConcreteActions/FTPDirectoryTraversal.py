from ipaddress import IPv4Address

from CybORG import Observation
from CybORG.Shared.Actions.ConcreteActions.ExploitAction import ExploitAction
from CybORG.Simulator import Host
from CybORG.Simulator import Process
from CybORG.Simulator import State


class FTPDirectoryTraversal(ExploitAction):
    def __init__(self, session: int, agent: str, ip_address: IPv4Address, target_session: int):
        super().__init__(session, agent, ip_address, target_session)

    def sim_execute(self, state: State) -> Observation:
        return self.sim_exploit(state, 21, 'femitter')

    def test_exploit_works(self, target_host: Host, vuln_proc: Process):
        # ask Max on what properties to check

        return bool(True)
