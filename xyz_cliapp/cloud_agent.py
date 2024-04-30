""" 
==========
CloudAgent
==========
@file_name: cloud_agent.py
@date: 2024-4-28
@author: BlackSheep Team, Netmind.AI
"""


import requests

from xyz.node.agent import Agent


class CloudAgent(Agent):
    def __init__(self):
        super().__init__()
        
        self.set_information({
            
        })
        
    def get_agent_api(self, agent_id):
        """
        """
        
        api_url = ""
        
        return api_url 
    
    def get_agent_api(self, api_url, **kwargs):
        """
        """
        
        # TODO: 设置子段的名字
        response = requests.post(f"{api_url}/add/", json=kwargs)
        print(response.json())
        
        return api_url
        
    



