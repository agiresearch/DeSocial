import time
import random
import numpy as np

class Coordinator:
    """
    Central coordinator for the distributed social recommendation system.
    Manages validator selection and consensus aggregation.
    """

    def __init__(self, num_node=100):
        """
        Initialize the coordinator.
        
        Parameters:
            num_node (int): Total number of nodes in the system.
        """
        self.num_node = num_node
        self.validator_list = []
        self.votes = []
        self.final_result = None
        
        print("Successfully created the coordination system.")
    
    def select_validators(self, val_tot, val_num):
        """
        Randomly select validators from the available pool.
        
        Parameters:
            val_tot (int): Total number of available validators.
            val_num (int): Number of validators to select.
            
        Returns:
            list: Indices of selected validators.
        """
        self.validator_list = random.sample(range(val_tot), min(val_num, val_tot))
        return self.validator_list
    
    def get_val_list(self):
        """
        Get the current list of selected validators.
        
        Returns:
            list: Current validator list.
        """
        return self.validator_list
    
    def reset_votes(self):
        """
        Reset the voting state for a new round.
        """
        self.votes = []
        self.final_result = None
    
    def collect_vote(self, validator_id, vote):
        """
        Collect a vote from a validator.
        
        Parameters:
            validator_id: ID of the voting validator.
            vote (bool): The validator's vote.
        """
        self.votes.append(vote)
    
    def finalize(self):
        """
        Aggregate votes using majority consensus.
        
        Returns:
            bool: The final decision based on majority voting.
        """
        if len(self.votes) == 0:
            self.final_result = False
        else:
            # Majority voting
            self.final_result = sum(self.votes) > len(self.votes) / 2
        return self.final_result
    
    def get_final_result(self):
        """
        Get the final aggregated result.
        
        Returns:
            bool: The final decision.
        """
        return self.final_result
