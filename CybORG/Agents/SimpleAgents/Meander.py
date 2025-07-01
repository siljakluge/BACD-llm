import random

from CybORG import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices
from CybORG.Shared.Actions.Action import Sleep
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os

class RedMeanderAgent(BaseAgent):
    # a red agent that meanders through scenario 1b
    def __init__(self):
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        self.observations = []
        self.llm_enabled = os.getenv('LLM_ENABLED')

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass
    
    #  This performs a breadth first search on all known hosts, scanning each one in turn, before attempting a mix of exploit and privilege escalate on the rest.
    
    def get_action(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        self._process_success(observation)
        print("OBS IN MEANDER FILE")
        print(observation) # last observation
        self.observations.append(observation)
        #print(action_space) is agentinterface
        session = list(action_space['session'].keys())[0]
        
        # Always impact if able
        if 'Op_Server0' in self.escalated_hosts:
            self.last_host = 'Op_Server0'
            return Impact(agent='Red', hostname='Op_Server0', session=session)

        # start by scanning all ip adresses in first subnet
        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue
            self.scanned_subnets.append(subnet)
            print("DRS")
            return DiscoverRemoteSystems(subnet=subnet, agent='Red', session=session)
        # scanning all ip adresses in this first subnet and discover network services (which means all ports)
        # # act on ip addresses discovered in first subnet
        addresses = [i for i in action_space["ip_address"]]
        random.shuffle(addresses)
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            self.scanned_ips.append(address)
            print("DNS")
            return DiscoverNetworkServices(ip_address=address, agent='Red', session=session)
        # priv esc on owned hosts
        hostnames = [x for x in action_space['hostname'].keys()] # this randomness !!! skip everything that is not in actionspace, start: user0 in action space
        random.shuffle(hostnames) 
        # Starts with User0 since Action_Space has this pre-determined and set to knowledge True
        # PE User 0, Interface to .186, Exploit .186 (works almost everytime) => if works: Sessions + System Info (User1), PE User 1
        # USER 1 (.186) LEAKS CONNECTION INFORMATION TO ENTERPRISE1 .158 after Privilige Escalate
        # DNS .158, Exploit .158 (does attack multiple times, wont work => works), PE on User4, DNS on Enterprise0, Exploit (two attacks => works), 
        # if attack on user xy did not work last time:
        #if self.last_host 
        for hostname in hostnames:
            # test if host is not known then continue
            if not action_space["hostname"][hostname]:
                continue
            # test if host is already priv esc
            if hostname in self.escalated_hosts:
                continue
            # test if host is exploited
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
            print("PE")
            return PrivilegeEscalate(hostname=hostname, agent='Red', session=session)

        #if self.llm_enabled:
        #    self.plan_attack(action_space=action_space)
        # access unexploited hosts
        for address in addresses:
            # test if output of observation matches expected output
            # imo: take one of the ips scanned and exploit
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            self.exploited_ips.append(address)
            self.last_ip = address
            print("Exploit")
            return ExploitRemoteService(ip_address=address, agent='Red', session=session)

        raise NotImplementedError('Red Meander has run out of options!')


    def plan_attack(self, action_space): 
        obs = self.observations # get ips, ports
        subnet_information =  "\n".join(str(d) for d in obs)

        print(subnet_information)
        red_agent_attacks = action_space["action"] # get all available actions, disable non-attack based
        red_agent_attacks[Sleep] = False
        red_agent_attacks[DiscoverNetworkServices] = False
        red_agent_attacks[DiscoverRemoteSystems] = False
        red_agent_attacks[ExploitRemoteService] = False
        enabled_attacks_str = "\n".join(attack.__name__ for attack, is_enabled in red_agent_attacks.items() if is_enabled)

        print(enabled_attacks_str)

        system_message_content = """You are an cybersecurity expert and answer the questions for cyber security attack strategies. You can only use the available scanned subnet and the attacks. \\ 
        Available Attacks:{attacks}\\
        Subnet Information:{subnet_information}""" # objectives
        system_message = SystemMessage(content=system_message_content)
        human_message = HumanMessage(content="What is best attack that is most likely to suceed?")
        ai_message = AIMessage(
            content="""\
        1. Choose a programming language: Decide on a programming language that you want to learn.

        2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

        3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
        """
        )
        return NotImplemented


    def call_llm_proxy(self, prompt):
        response = input()
        return response

    def _process_success(self, observation):
        if self.last_ip is not None:
            if observation['success'] == True:
                self.host_ip_map[[value['System info']['Hostname'] for key, value in observation.items()
                                  if key != 'success' and 'System info' in value
                                  and 'Hostname' in value['System info']][0]] = self.last_ip
            else:
                self._process_failed_ip()
            self.last_ip = None
        if self.last_host is not None:
            if observation['success'] == False:
                if self.last_host in self.escalated_hosts:
                    self.escalated_hosts.remove(self.last_host)
                if self.last_host in self.host_ip_map and self.host_ip_map[self.last_host] in self.exploited_ips:
                    self.exploited_ips.remove(self.host_ip_map[self.last_host])
            self.last_host = None

    def _process_failed_ip(self):
        self.exploited_ips.remove(self.last_ip)
        hosts_of_type = lambda y: [x for x in self.escalated_hosts if y in x]
        if len(hosts_of_type('Op')) > 0:
            for host in hosts_of_type('Op'):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)
        elif len(hosts_of_type('Ent')) > 0:
            for host in hosts_of_type('Ent'):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)

    def end_episode(self):
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None

    def set_initial_values(self, action_space, observation):
        pass
