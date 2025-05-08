from web3 import Web3
import numpy as np
from utils.DataLoader import Data

# global lists, storing all the role objects.
user_storage = []
address_id_map = {}

class BC_User:

    def __init__(self, user_id, bc_address, num_of_val, url, web3):
        """
            Here gives a brief description of the arguments.
            Web3 Properties:
                bc_address: the address of the user on the blockchain.
                num_of_val: the number of validators in the system.
                url: the url of the blockchain.
                web3: the web3 object.
                contract: the smart contract object.
            
            Local Data Storage:
                user_id: the id of the user.
                model: the model object.
                train_data: the training data.
                val_data: the validation data.
                test_data: the test data.
                requests_collected: the requests collected from the source nodes.
        """

        ## Web3 Properties
        self.bc_address = bc_address
        self.url = url
        self.web3 = web3
        self.contract = None # will initialize in run.py
        self.num_of_val = num_of_val
        
        ## Local Data Storage
        self.user_id = user_id
        self.model = None # will initialize in run.py
        self.optimizer = None # will initialize in run.py
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.requests_collected = []
    
    def send_a_request(self, target, inter_terminal, timestamp):
        """
            The initiator sends a request to the blokchain that it wants to add an edge to the target node.
            Parameters:
                target: the target node.
                inter_terminal: the intermediate terminal node.
                timestamp: the timestamp of the request.
        """
        # the user acts as an initiator, sending a request to solidity by `request(target, inter_terminal)`
        # in request function, the smart contract collects the request.
        tx = self.contract.functions.request(user_storage[target].bc_address, inter_terminal).transact({
            "from": self.bc_address,
            "gas": 300000,
            "gasPrice": self.web3.to_wei("20", "gwei"), # to make the transaction more efficient, set the gas price to 20 gwei
        })
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        # the smart contract recognized this request will be stored in the intermediate terminal node.
        # the intermediate terminal node will store the request in its local data storage.
        if tx_receipt["status"] == 1:
            # if the transaction is successful, the request is stored in the intermediate terminal node.
            user_storage[inter_terminal].requests_collected.append([self.user_id, target, timestamp])

    def retrieve_test_data(self, inter_terminal):
        """
            Signed by the validator.
            The validator retrieves the test data from the intermediate terminal node.
            Parameters:
                inter_terminal: the intermediate terminal node.
        """
        tx = self.contract.functions.broadcast(inter_terminal).transact({
            "from": self.bc_address,
            "gas": 300000,
            "gasPrice": self.web3.to_wei("20", "gwei")
        })
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        if tx_receipt["status"] == 1:
            # if the transaction is successful, the validator copies the test data from the intermediate terminal node.
            self.test_data = user_storage[inter_terminal].test_data

    def broadcast(self, inter_terminal):
        """
            Signed by the receiver.
            After validating all the tasks in predicting the next preiod, all the ground truth edges will be broadcast to everyone, preparing for the next period's experiment.
            Step:
                1. the receiver signs the retrieval.
                2. the receiver copies the ground truth edges to its local data storage.
        """
        # broadcast的逻辑，看看这个输入的参数，是不是smart contract里存的inter_terminal，不允许乱拿
        tx = self.contract.functions.broadcast(inter_terminal).transact({
            "from": self.bc_address,
            "gas": 300000,
            "gasPrice": self.web3.to_wei("20", "gwei")
        })
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        if tx_receipt["status"] == 1:
            # user_storage[target].test_data
            self.update_social_network()
            # if the transaction is successful, the ground truth edges are broadcasted to everyone.
            self.test_data = user_storage[inter_terminal].test_data

    def select_validators(self, val_tot, val_num):
        """
            the intermediate terminal node request the blockchain to choose validator randomly.
            Parameters:
                val_tot: the total number of validators in a specified backbone community.
                val_num: the number of validators to be chosen.
            Return:
                validators: the list of validator indices.
        """
        validators = []
        tx = self.contract.functions.select_validators(val_tot, val_num).transact({
            "from": self.bc_address,
            "gas": 300000,
            "gasPrice": self.web3.to_wei("20", "gwei")
        })
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        if tx_receipt["status"] == 1:
            validators = self.contract.functions.get_validators().call()

        return validators

    def vote(self, agree):
        """
            Each validator give their votes based on their backbone.
        """
        tx = self.contract.functions.vote(self.bc_address, agree).transact({
            "from": self.bc_address,
            "gas": 300000,
            "gasPrice": self.web3.to_wei("20", "gwei")
        })
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)

    def update_social_network(self):
        """
            Two steps:
                1. merge the old train data and the old val data to new train data.
                2. the new val data is the test data.
        """
        src_node_ids = np.concatenate((self.train_data.src_node_ids, self.val_data.src_node_ids), axis=0)
        dst_node_ids = np.concatenate((self.train_data.dst_node_ids, self.val_data.dst_node_ids), axis=0)
        node_interact_times = np.concatenate((self.train_data.node_interact_times, self.val_data.node_interact_times), axis=0)
        self.train_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
        self.val_data = self.test_data