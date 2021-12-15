__author__ = "Chunpai W."
__email__ = "cwang25@albany.edu"

import pickle
import random

import numpy as np
from numpy.random import RandomState

from mochi.utils import *
from scipy.stats import spearmanr


def eval_policy(policy_name, top_k, users, items, user_records, pretest_posttest_dict,
                next_item_distr_dict):
    traj_dch_list = []
    traj_ipw_dch_list = []
    traj_score_growth_list = []
    traj_posttest_score_list = []
    # sample a test user
    for traj, user_name in enumerate(users):
        user_q_list = user_records[user_name]
        pretest, posttest = pretest_posttest_dict[user_name]
        score_growth = posttest - pretest
        traj_score_growth_list.append(score_growth)
        traj_posttest_score_list.append(posttest)

        dch = []
        ipw_dch = []
        for t in range(len(user_records[user_name])):
            # generate an action that follows policy (for BKT-MP, it is based on curr_bel)
            # once an action is generated, it is removed from the action space (problem_list)

            if policy_name == "random":
                candidates = random.sample(items, top_k)
            else:
                raise NotImplementedError

            if t + 1 < len(user_q_list):
                current_q = user_q_list[t]
                true_next_question = user_q_list[t + 1]
                if true_next_question in candidates:
                    candidates = list(candidates)
                    prBlue("HIT: student {}".format(user_name))
                    print("candidates {}".format(candidates))
                    print("true next question: {}".format(true_next_question))
                    print("true next question distribution (next-q: count): {}".format(
                        next_item_distr_dict[current_q]))
                    ndch = 1.0 / np.log2(candidates.index(true_next_question) + 1 + 1)
                    total_count = 0.
                    for question in next_item_distr_dict[current_q].keys():
                        total_count += next_item_distr_dict[current_q][question]

                    propensity_score = next_item_distr_dict[current_q][
                                           true_next_question] / total_count
                    propensity_ndch = ndch / propensity_score
                    prGreen("ndch score {} , propensity score: {}, propensity ndch score {}".format(
                        ndch, propensity_score, propensity_ndch))
                    print('-' * 20)
                    ipw_dch.append(propensity_ndch)
                    dch.append(ndch)
                else:
                    prRed("MISS: student {}".format(user_name))
                    print("candidates {}".format(candidates))
                    print("true next question: {}".format(true_next_question))
                    print("true next question distribution (next-q: count): {}".format(
                        next_item_distr_dict[current_q]))
                    print('-' * 20)
                    dch.append(0.)
                    ipw_dch.append(0.)
        traj_dch_list.append(dch)
        traj_ipw_dch_list.append(ipw_dch)

    avg_dch_list = []
    avg_ipw_dch_list = []
    for ipw_dch in traj_ipw_dch_list:
        avg_ipw_dch_list.append(np.mean(ipw_dch))
    for dch in traj_dch_list:
        avg_dch_list.append(np.mean(dch))
    print("average dch: {}".format(list(np.round(avg_dch_list, 2))))
    print("average ipw dch: {}".format(list(np.round(avg_ipw_dch_list, 2))))
    print("post test score: {}".format(list(np.round(traj_posttest_score_list, 2))))
    print("score growth: {}".format(list(np.round(traj_score_growth_list, 2))))
    correlation_dch_post_score = spearmanr(avg_dch_list, traj_posttest_score_list)
    correlation_dch_score_growth = spearmanr(avg_dch_list, traj_score_growth_list)

    correlation_ipw_dch_post_score = spearmanr(avg_ipw_dch_list, traj_posttest_score_list)
    correlation_ipw_dch_score_growth = spearmanr(avg_ipw_dch_list, traj_score_growth_list)

    prRed("top-k: {}, policy: {}".format(top_k, policy))
    print("dch vs post scores spearman correlation: {} {}".format(
        correlation_dch_post_score.correlation, correlation_dch_post_score.pvalue))
    print("dch vs score growth spearman correlation: {} {}".format(
        correlation_dch_score_growth.correlation, correlation_dch_score_growth.pvalue))
    print("")
    print("ipw-dch vs post scores spearman correlation: {} {}".format(
        correlation_ipw_dch_post_score.correlation, correlation_ipw_dch_post_score.pvalue))
    print("ipw-dch vs score growth spearman correlation: {} {}".format(
        correlation_ipw_dch_score_growth.correlation, correlation_ipw_dch_score_growth.pvalue))
    print("")
    print("{} {}".format(correlation_dch_post_score.correlation, correlation_dch_post_score.pvalue))
    print("{} {}".format(correlation_dch_score_growth.correlation,
                         correlation_dch_score_growth.pvalue))
    print("{} {}".format(correlation_ipw_dch_post_score.correlation,
                         correlation_ipw_dch_post_score.pvalue))
    print("{} {}".format(correlation_ipw_dch_score_growth.correlation,
                         correlation_ipw_dch_score_growth.pvalue))
    print("avg-dch  ", *np.round(avg_dch_list, 4))
    print("avg-ipw-dch  ", *np.round(avg_ipw_dch_list, 4))
    print("post-scores  ", *traj_posttest_score_list)
    print("score-growth  ", *np.round(traj_score_growth_list, 4))
    return avg_dch_list, avg_ipw_dch_list, traj_posttest_score_list, traj_score_growth_list, \
           correlation_dch_post_score, correlation_dch_score_growth, \
           correlation_ipw_dch_post_score, correlation_ipw_dch_score_growth


def get_data(data_str):
    data = pickle.load(open("../data/{}/data.pkl".format(data_str), "rb"))
    user_list = data["users"]
    item_list = data['items']
    user_records_dict = data["records"]
    pretest_posttest_dict = data["pretest_posttest_dict"]
    next_item_distr_dict = data["next_items_dict"]
    return user_list, item_list, user_records_dict, pretest_posttest_dict, next_item_distr_dict


if __name__ == '__main__':
    data_str = "MasteryGrids"
    top_k = 3
    policy = "random"
    sample_size = 1

    users, items, user_records, pretest_posttest_dict, next_item_distr_dict = get_data(data_str)

    if policy != "random":
        eval_policy(policy, top_k, users, items, user_records, pretest_posttest_dict,
                    next_item_distr_dict)
    else:
        traj_lengths_list = []
        avg_dch_list_list = []
        avg_ipw_dch_list_list = []
        traj_posttest_score_list_list = []
        traj_score_growth_list_list = []
        final_rewards_list = []
        correlation_dch_post_score_list = []
        pvalue_dch_post_score_list = []
        correlation_dch_traj_length_list = []
        pvalue_dch_traj_length_list = []
        correlation_dch_score_growth_list = []
        pvalue_dch_score_growth_list = []
        correlation_dch_simulated_reward_list = []
        pvalue_dch_simulated_reward_list = []
        correlation_ipw_dch_post_score_list = []
        pvalue_ipw_dch_post_score_list = []
        correlation_ipw_dch_traj_length_list = []
        pvalue_ipw_dch_traj_length_list = []
        correlation_ipw_dch_score_growth_list = []
        pvalue_ipw_dch_score_growth_list = []
        correlation_ipw_dch_simulated_reward_list = []
        pvalue_ipw_dch_simulated_reward_list = []

        for i in range(sample_size):
            outputs = eval_policy(policy, top_k, users, items, user_records,
                                  pretest_posttest_dict, next_item_distr_dict)
            avg_dch_list_list.append(outputs[0])
            avg_ipw_dch_list_list.append(outputs[1])
            traj_posttest_score_list_list.append(outputs[2])
            traj_score_growth_list_list.append(outputs[3])
            correlation_dch_post_score_list.append(outputs[4].correlation)
            pvalue_dch_post_score_list.append(outputs[4].pvalue)
            correlation_dch_score_growth_list.append(outputs[5].correlation)
            pvalue_dch_score_growth_list.append(outputs[5].pvalue)
            correlation_ipw_dch_post_score_list.append(outputs[6].correlation)
            pvalue_ipw_dch_post_score_list.append(outputs[6].pvalue)
            correlation_ipw_dch_score_growth_list.append(outputs[7].correlation)
            pvalue_ipw_dch_score_growth_list.append(outputs[7].pvalue)

        print("")
        print("{} {}".format(np.mean(correlation_dch_post_score_list),
                             np.mean(pvalue_dch_post_score_list)))
        print("{} {}".format(np.mean(correlation_dch_score_growth_list),
                             np.mean(pvalue_dch_score_growth_list)))
        print("{} {}".format(np.mean(correlation_ipw_dch_post_score_list),
                             np.mean(pvalue_ipw_dch_post_score_list)))
        print("{} {}".format(np.mean(correlation_ipw_dch_score_growth_list),
                             np.mean(pvalue_ipw_dch_score_growth_list)))
        print("avg-dch  ", *np.round(np.mean(avg_dch_list_list, axis=0), 4))
        print("avg-ipw-dch  ", *np.round(np.mean(avg_ipw_dch_list_list, axis=0), 4))
        print("post-scores  ", *np.mean(traj_posttest_score_list_list, axis=0))
        print("score-growth  ", *np.round(np.mean(traj_score_growth_list_list, axis=0), 4))
