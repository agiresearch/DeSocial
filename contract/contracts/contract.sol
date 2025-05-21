// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DeSocial {

    uint256 public agree_count;
    uint256 public disagree_count;
    bool public FINAL_RESULT;
    int256 public intermediate_terminal;

    int256[] public val_list;

    mapping(address => bool) public unique;

    event DebugLog(address indexed user, uint256 balance, int256 tau);
    event vote_casted(address indexed voter, bool agree);

    event Request(address indexed requester, address indexed target, int256 inter_terminal);

    function set_inter_terminal(int256 upload_num) public {
        intermediate_terminal = upload_num;
    }

    function set_val_list(int256[] memory upload_list) public {
        val_list = upload_list;
    }

    function get_val_list() public view returns (int256[] memory) {
        return val_list;
    }

    function set_FINAL_RESULT(bool result) public {
        FINAL_RESULT = result;
    }

    function get_FINAL_RESULT() public view returns (bool){
        return FINAL_RESULT;
    }

    function vote(address voter, bool agree) public payable {
        if (agree) {
            agree_count++;
        } else {
            disagree_count++;
        }
        emit vote_casted(voter, agree);
    }

    function finalize() public returns (bool) {
        bool result = agree_count > disagree_count;
        agree_count = 0;
        disagree_count = 0;
        set_FINAL_RESULT(result);
        return result;
    }

    function select_validators(int256 val_tot, int256 val_num) public returns (int256[] memory) {
        // generate a list of validators based on the total number of validators and the number of validators to select
        // by generating val_num random numbers between 0 and val_tot-1, without replacement
        require(val_tot > 0 && val_num > 0, "invalid input");
        require(val_num <= val_tot, "cannot select more than total");

        int256[] memory selected = new int256[](uint256(val_num));
        bool[] memory used = new bool[](uint256(val_tot));
        uint256 count = 0;

        while (count < uint256(val_num)) {
            int256 rand = int256(uint256(keccak256(abi.encodePacked(block.timestamp, block.prevrandao, block.number, msg.sender, count))) % uint256(val_tot));
            if (!used[uint256(rand)]) {
                selected[count] = rand;
                used[uint256(rand)] = true;
                count++;
            }
        }

        agree_count = 0;
        disagree_count = 0;
        set_val_list(selected);
        return selected;
    }

    function request(address target, int256 inter_terminal) public payable {
        // Emit an event to notify that a request has been made
        set_inter_terminal(inter_terminal);
        emit Request(msg.sender, target, inter_terminal);
    }

    function retrieve(int256 inter_terminal) public view {
        // the smart contract is supposed to check whether the intermediate terminal is correct
        require(inter_terminal == intermediate_terminal, "Invalid intermediate terminal");
    }

}
