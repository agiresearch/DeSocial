// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Web3SN {
    struct vote_str {
        address voter;
        bool agree;
    }

    uint256 public agree_count;
    uint256 public disagree_count;
    bool public FINAL_RESULT;
    vote_str[] public voting_list;

    address[] public initiator_only_vis_list;
    address[] public target_only_vis_list;
    address[] public both_vis_list;
    address[] public none_repu_vis_list;
    address[] public val_list_offchain;

    mapping(address => bool) public unique;

    event DebugLog(address indexed user, uint256 balance, int256 tau);
    event vote_casted(address indexed voter, bool agree);
    event validation_started(address indexed initiator, address target, string val_type);
    event Promote(address indexed promoter, address indexed target, uint256 fee);
    event Subscribe(address indexed subscriber, address indexed target, uint256 fee);

    // on-off chain data transition
    function get_vis_initiator_only() public view returns (address[] memory) {
        return initiator_only_vis_list;
    }

    function get_vis_target_only() public view returns (address[] memory) {
        return target_only_vis_list;
    }

    function get_vis_both() public view returns (address[] memory) {
        return both_vis_list;
    }

    function get_vis_none_repu() public view returns (address[] memory) {
        return none_repu_vis_list;
    }

    function get_val_list() public view returns (address[] memory){
        return val_list_offchain;
    }

    function get_FINAL_RESULT() public view returns (bool){
        return FINAL_RESULT;
    }

    function set_vis_initiator_only(address[] memory upload_list) public {
        initiator_only_vis_list = upload_list;
    }

    function set_vis_target_only(address[] memory upload_list) public {
        target_only_vis_list = upload_list;
    }

    function set_vis_both(address[] memory upload_list) public {
        both_vis_list = upload_list;
    }

    function set_vis_none_repu(address[] memory upload_list) public {
        none_repu_vis_list = upload_list;
    }

    function set_val_list(address[] memory val_list) public {
        val_list_offchain = val_list;
    }

    function set_FINAL_RESULT(bool result) public {
        FINAL_RESULT = result;
    }

    function verification(int256 tau, uint256 limit_len) public returns (address[] memory) {
        address[] memory validators = new address[](limit_len);
        uint256 count = 0;

        for (uint256 i = 0; i < both_vis_list.length; i++) {
            //emit DebugLog(vis_both[i], vis_both[i].balance, tau);
            if (both_vis_list[i].balance >= uint256(tau)) {
                validators[count] = both_vis_list[i];
                count++;
                if (count == limit_len){
                    return validators;
                }
            }
        }

        for (uint256 i = 0; i < target_only_vis_list.length; i++) {
            //emit DebugLog(vis_target[i], vis_target[i].balance, tau);
            if (target_only_vis_list[i].balance >= uint256(tau)) {
                validators[count] = target_only_vis_list[i];
                count++;
                if (count == limit_len){
                    return validators;
                }
            }
        }

        for (uint256 i = 0; i < initiator_only_vis_list.length; i++) {
            //emit DebugLog(vis_initiator[i], vis_initiator[i].balance, tau);
            if (initiator_only_vis_list[i].balance >= uint256(tau)) {
                validators[count] = initiator_only_vis_list[i];
                count++;
                if (count == limit_len){
                    return validators;
                }
            }
        }

        for (uint256 i = 0; i < none_repu_vis_list.length; i++) {
            //emit DebugLog(vis_none_repu[i], vis_none_repu[i].balance, tau);
            if (none_repu_vis_list[i].balance >= uint256(tau)) {
                validators[count] = none_repu_vis_list[i];
                count++;
                if (count == limit_len){
                    return validators;
                }
            }
        }
        
        set_val_list(validators);

        return validators;
    }
    function vote(address voter, bool agree) public payable {
        if (agree) {
            agree_count++;
        } else {
            disagree_count++;
        }
        voting_list.push(vote_str({voter: voter, agree: agree}));
        emit vote_casted(voter, agree);
    }

    function finalize() public returns (bool) {
        bool result = agree_count > disagree_count;
        uint256 winner_num = result ? agree_count : disagree_count;
        
        uint256 reward = address(this).balance / winner_num;

        for (uint256 i = 0; i < voting_list.length; i++) {
            address voter = voting_list[i].voter;
            if (voting_list[i].agree == result) {
                payable(voter).transfer(reward);
            }
        }
        
        delete voting_list;
        agree_count = 0;
        disagree_count = 0;
        set_FINAL_RESULT(result);

        return result;
    }

    function promote(address target, uint256 promotion_fee) public payable {
        require(address(msg.sender).balance >= uint256(promotion_fee), "Insufficient balance");

        payable(target).transfer(promotion_fee);
        emit Promote(msg.sender, target, promotion_fee);
    }

    function subscribe(address target, uint256 subscription_fee) public payable {
        require(address(msg.sender).balance >= uint256(subscription_fee), "Insufficient balance");

        payable(target).transfer(subscription_fee);
        emit Subscribe(msg.sender, target, subscription_fee);
    }

    function start_validation(address initiator, address target, string memory val_type, int256 tau, uint256 limit_len) public payable returns (address[] memory) {
        delete voting_list;
        agree_count = 0;
        disagree_count = 0;
        
        address[] memory val_list = verification(tau, limit_len);
        set_val_list(val_list);

        emit validation_started(initiator, target, val_type);
        return val_list;
    }
}
