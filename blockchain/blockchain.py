import requests
from web3 import Web3
from web3.exceptions import ContractLogicError
import json
import time
from tqdm import tqdm
import random
import subprocess

# https://ethereum.org/en/developers/docs/programming-languages/python/
class Blockchain:

    """
        `__init__(self, provider_url, contract_json_path, account_number)`  
        **Description:** Creates a blockchain, with specified number of accounts and initial number of tokens in each account. Also deploys the smart contract (which contains the verification algorithm).  
        **Input:**  
        - `provider_url` (`str`): URL of the provider.  
        - `contract_json_path` (`str`): Path to the smart contract.  
        - `account_number` (`int`): Total number of accounts.  

        **Return:** `None`  
    """
    def __init__(self, provider_url, contract_json_path, num_node = 100):
        self.num_node = num_node
        try:
            self.url = provider_url
            self.web3 = Web3(Web3.HTTPProvider(provider_url))
            if not self.web3.is_connected():
                raise ConnectionError("Failed to connect to the blockchain.")
            else:
                print("The blockchain system exists.")
        except:
            self.reset_blockchain()
            self.url = provider_url
            self.web3 = Web3(Web3.HTTPProvider(provider_url))

        print("Successfully created a blockchain system.")

        try:
            self.publish_contract(contract_json_path)
            print("Successfully published a smart contract (concensus of decentralized social network).")

        except:
            print("Can't find the smart contract address, please config it with a valid json file.")
    
    """
        `reset_blockchain(self, num_token)`  
            **Description:** To ensure absolute fairness during initialization in experiments, all tokens are moved to a specific account and then redistributed as `new_account_balance` tokens from that account (refer to `create_one_account(self)`).  
            **Input:** `num_token` (float): token per node initially.
            **Return:** `None`  
    """
    def reset_blockchain(self, num_token = 1000):
        command = f"ganache --accounts {self.num_node} --defaultBalanceEther {num_token} --port 7545"
        self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ganache_pid = self.process.pid
        print(f"Ganche is running with PID: {ganache_pid}")
        print("Reseting the blockchain...")
        time.sleep(5)
    
    """
        `print_account_list(self)`  
            **Description:** Prints the current account list along with their balances.  
            **Input:** `None`  
            **Return:** `None`  
    """
    def print_account_list(self):
        accounts = self.web3.eth.accounts
        for i in range(len(accounts)):
            balance = self.web3.eth.get_balance(accounts[i])
            balance = self.web3.from_wei(balance, 'ether')
            print(f"Account: {accounts[i]}    Balance: {balance} ETH")
    
    """
        `publish_contract(self, contract_json_path)`  
            **Description:** Deploys the smart contract.  
            **Input:** `contract_json_path` (`str`): the json path of the smart contract file representation.
            **Return:** `None`
    """
    def publish_contract(self, contract_json_path):
        # Get the ABI (Application Binary Interface) of the smart contract. python and solidity connect through the ABI.
        # Compiling smart contracts into abi json files requires truffle (a smart contract compilation environment)
        _ = self.web3.eth.default_account = self.web3.eth.accounts[0]
        with open(contract_json_path, "r") as f:
            contract_json = json.load(f)
            abi = contract_json["abi"]
            bytecode = contract_json["bytecode"]
            self.contract_abi = abi
            self.contract_bytecode = bytecode

        # Publish contract
        self.contract = self.web3.eth.contract(abi = self.contract_abi, bytecode = self.contract_bytecode)
        tx_hash = self.contract.constructor().transact()
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

        # Get contract address
        self.contract_address = tx_receipt.contractAddress
        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.contract_abi)
        print(f"Deployed smart contract at address {self.contract_address}.")
