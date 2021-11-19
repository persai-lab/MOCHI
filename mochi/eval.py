__author__ = "Chunpai W."
__email__ = "cwang25@albany.edu"

import pickle
import numpy as np
from numpy.random import RandomState

from mochi.utils import *
from scipy.stats import spearmanr


def eval_policy(policy_name, top_k, user_list, pretest_and_posttest_dict, next_item_distribution_dict):
    traj_dch_list = []
    traj_ipw_dch_list = []
    traj_score_growth_list = []
    traj_posttest_score_list = []
    # sample a test user
    for traj, user_name in enumerate(user_list):
        # reset the test student models' initial belief when new trajectory is sampled as the
        # training user's initial belief
        pretest = pretest_list[traj]
        user_q_list = users_question_list[traj]
        user_a_list = users_answer_list[traj]
        pretest, posttest = pretest_and_posttest_dict[user_name]
        score_growth = posttest - pretest
        traj_score_growth_list.append(score_growth)

        traj_posttest_score_list.append(posttest)
        prPurple("\ntraj:{}/{}, user:{}, pretest:{}, posttest:{}".format(
            traj, len(users_name_list), user_name, pretest, posttest))

        q_list = []
        a_list = []
        dch = []
        ipw_dch = []
        for t in range(traj_lengths[traj] if type(traj_lengths) == list else traj_lengths):
            # generate an action that follows policy (for BKT-MP, it is based on curr_bel)
            # once an action is generated, it is removed from the action space (problem_list)

            if policy == "baseline":
                true_question = user_q_list[t]
                idx = problem_seq.index(true_question)
                if idx + 1 != len(problem_seq):
                    question = problem_seq[idx + 1]
                    candidates = problem_seq[idx + 1: idx + top_k + 1]
                else:
                    question = problem_seq[-1]
                    candidates = problem_seq[-top_k:][::-1]
            else:
                candidates = dkt_agent.plan(policy)
                print(candidates)
                question = candidates[0]

            if t + 1 < len(user_q_list):
                current_q = user_q_list[t]
                current_score = user_a_list[t]
                true_next_question = user_q_list[t + 1]
                if true_next_question in candidates:
                    candidates = list(candidates)
                    prBlue("HIT: student {}".format(user_name))
                    print("current question and score: ({}, {:.3f})".format(
                        current_q, current_score))
                    print("candidates {}".format(candidates))
                    print("true next question: {}".format(true_next_question))
                    print("true next question distribution (next-q: count): {}".format(
                        next_item_distribution_dict[current_q]))
                    ndch = 1.0 / np.log2(candidates.index(true_next_question) + 1 + 1)
                    total_count = 0.
                    for question in next_item_distribution_dict[current_q].keys():
                        total_count += next_item_distribution_dict[current_q][question]

                    propensity_score = next_item_distribution_dict[current_q][
                                           true_next_question] / total_count
                    propensity_ndch = ndch / propensity_score
                    prGreen("ndch score {} , propensity score: {}, propensity ndch score {}".format(
                        ndch, propensity_score, propensity_ndch))
                    print('-' * 20)
                    ipw_dch.append(propensity_ndch)
                    dch.append(ndch)
                else:
                    prRed("MISS: student {}".format(user_name))
                    print(
                        "current question and score: ({}, {:.3f})".format(current_q, current_score))
                    print("candidates {}".format(candidates))
                    print("true next question: {}".format(true_next_question))
                    print("true next question distribution (next-q: count): {}".format(
                        next_item_distribution_dict[current_q]))
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





if __name__ == '__main__':
    fold = 1
    top_k = 1

    policy = "random"

    if policy != "random":
        eval_policy(policy, top_k, pretest_and_posttest_dict, next_item_distribution_dict)
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

        for i in range(100):
            outputs = eval_policy(policy, top_k, pretest_and_posttest_dict, next_item_distribution_dict)
            traj_lengths_list.append(outputs[0])
            avg_dch_list_list.append(outputs[1])
            avg_ipw_dch_list_list.append(outputs[2])
            traj_posttest_score_list_list.append(outputs[3])
            traj_score_growth_list_list.append(outputs[4])
            final_rewards_list.append(outputs[5])
            correlation_dch_post_score_list.append(outputs[6].correlation)
            pvalue_dch_post_score_list.append(outputs[6].pvalue)
            correlation_dch_traj_length_list.append(outputs[7].correlation)
            pvalue_dch_traj_length_list.append(outputs[7].pvalue)
            correlation_dch_score_growth_list.append(outputs[8].correlation)
            pvalue_dch_score_growth_list.append(outputs[8].pvalue)
            correlation_dch_simulated_reward_list.append(outputs[9].correlation)
            pvalue_dch_simulated_reward_list.append(outputs[9].pvalue)
            correlation_ipw_dch_post_score_list.append(outputs[10].correlation)
            pvalue_ipw_dch_post_score_list.append(outputs[10].pvalue)
            correlation_ipw_dch_traj_length_list.append(outputs[11].correlation)
            pvalue_ipw_dch_traj_length_list.append(outputs[11].pvalue)
            correlation_ipw_dch_score_growth_list.append(outputs[12].correlation)
            pvalue_ipw_dch_score_growth_list.append(outputs[12].pvalue)
            correlation_ipw_dch_simulated_reward_list.append(outputs[13].correlation)
            pvalue_ipw_dch_simulated_reward_list.append(outputs[13].pvalue)

        print("")
        print("{} {}".format(np.mean(correlation_dch_post_score_list),
                             np.mean(pvalue_dch_post_score_list)))
        print("{} {}".format(np.mean(correlation_dch_traj_length_list),
                             np.mean(pvalue_dch_traj_length_list)))
        print("{} {}".format(np.mean(correlation_dch_score_growth_list),
                             np.mean(pvalue_dch_score_growth_list)))
        print("{} {}".format(np.mean(correlation_dch_simulated_reward_list),
                             np.mean(pvalue_dch_simulated_reward_list)))
        print("{} {}".format(np.mean(correlation_ipw_dch_post_score_list),
                             np.mean(pvalue_ipw_dch_post_score_list)))
        print("{} {}".format(np.mean(correlation_ipw_dch_traj_length_list),
                             np.mean(pvalue_ipw_dch_traj_length_list)))
        print("{} {}".format(np.mean(correlation_ipw_dch_score_growth_list),
                             np.mean(pvalue_ipw_dch_score_growth_list)))
        print("{} {}".format(np.mean(correlation_ipw_dch_simulated_reward_list),
                             np.mean(pvalue_ipw_dch_simulated_reward_list)))
        print("traj.-length  ", *(np.mean(traj_lengths_list, axis=0)))
        print("avg-dch  ", *np.round(np.mean(avg_dch_list_list, axis=0), 4))
        print("avg-ipw-dch  ", *np.round(np.mean(avg_ipw_dch_list_list, axis=0), 4))
        print("post-scores  ", *np.mean(traj_posttest_score_list_list, axis=0))
        print("score-growth  ", *np.round(np.mean(traj_score_growth_list_list, axis=0), 4))
        print("rewards  ", *np.round(np.mean(final_rewards_list, axis=0), 4))
