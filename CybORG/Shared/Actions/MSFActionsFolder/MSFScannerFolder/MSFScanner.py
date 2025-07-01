# Copyright DST Group. Licensed under the MIT license.
from CybORG import MSFAction
from CybORG.Simulator import State


class MSFScanner(MSFAction):
    def __init__(self, session, agent):
        super().__init__(session, agent)

    def sim_execute(self, state: State):
        pass
