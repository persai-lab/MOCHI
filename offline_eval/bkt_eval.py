__author__ = "Chunpai W."
__email__ = "cwang25@albany.edu"

# policies are built based on BKT plan model, which is used to trace student's knowledge based on
# historical performance.


import argparse
import pickle
import numpy as np
from numpy.random import RandomState

from offline_eval.bkt import getYudelsonModel, compute_bkt_last_belief
from offline_eval.bkt import BKTModel
from offline_eval.rewards import LogisticModel, LinearModel
from offline_eval.utils import *
from scipy.stats import spearmanr


def eval_policy(plan_model, policy, traj_lengths, pretest_list, users_question_list,
                users_answer_list, users_name_list, next_questions_dict, top_k=1):
    prBlue("Policy: {}, Number of Test Users: {}".format(policy, len(users_name_list)))
    # if given a list of trajectory lengths, sample numTraj trajectory lengths with replacement.

    traj_dch_list = []
    traj_ipw_dch_list = []
    traj_score_growth_list = []
    traj_posttest_score_list = []
    # sample a test user
    for traj, user_name in enumerate(users_name_list):
        # reset the test student models' initial belief when new trajectory is sampled as the
        # training user's initial belief
        pretest = pretest_list[traj]
        user_q_list = users_question_list[traj]
        user_a_list = users_answer_list[traj]
        pretest, posttest = user_name_pretest_posttest_scores[user_name]
        score_growth = posttest - pretest
        traj_score_growth_list.append(score_growth)

        traj_posttest_score_list.append(posttest)
        prPurple("\ntraj:{}/{}, user:{}, pretest:{}, posttest:{}".format(
            traj, len(users_name_list), user_name, pretest, posttest))
        plan_model.resetState(pretest)  # for plan model, we don't reset the pretest score

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
                candidates = plan_model.plan(policy)
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
                        next_questions_dict[current_q]))
                    ndch = 1.0 / np.log2(candidates.index(true_next_question) + 1 + 1)
                    total_count = 0.
                    for question in next_questions_dict[current_q].keys():
                        total_count += next_questions_dict[current_q][question]

                    propensity_score = next_questions_dict[current_q][
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
                        next_questions_dict[current_q]))
                    print('-' * 20)
                    dch.append(0.)
                    ipw_dch.append(0.)

            true_q = user_q_list[t]
            true_a = user_a_list[t]
            topic_name = problem_topic_mapping[true_q]
            step_outcomes = [(true_q, true_a, 0)]
            q_list.append(true_q)
            a_list.append(true_a)
            plan_model.updateState(step_outcomes)
            curr_belief = list(np.round(plan_model.curr_bel[:, 1], 4))
            kc_id = plan_model.kcMap[topic_name]
            print("kc id: {}, current updated belief: ".format(kc_id), end="")
            for index, bel in enumerate(curr_belief):
                if index == kc_id:
                    print("{} ".format(strGreen(bel)), end="")
                else:
                    print("{} ".format(bel), end="")
        traj_dch_list.append(dch)
        traj_ipw_dch_list.append(ipw_dch)
        plan_model.updateReward()  # collect reward of curr. user into plan_model.finalRewards

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
    correlation_dch_traj_length = spearmanr(avg_dch_list, traj_lengths)
    correlation_dch_score_growth = spearmanr(avg_dch_list, traj_score_growth_list)
    correlation_dch_simulated_reward = spearmanr(avg_dch_list, plan_model.finalRewards)

    correlation_ipw_dch_post_score = spearmanr(avg_ipw_dch_list, traj_posttest_score_list)
    correlation_ipw_dch_traj_length = spearmanr(avg_ipw_dch_list, traj_lengths)
    correlation_ipw_dch_score_growth = spearmanr(avg_ipw_dch_list, traj_score_growth_list)
    correlation_ipw_dch_simulated_reward = spearmanr(avg_ipw_dch_list, plan_model.finalRewards)

    prRed("top-k: {}, policy: {}".format(top_k, policy))
    print("dch vs post scores spearman correlation: {} {}".format(
        correlation_dch_post_score.correlation, correlation_dch_post_score.pvalue))
    print("dch vs traj length spearman correlation: {} {}".format(
        correlation_dch_traj_length.correlation, correlation_dch_traj_length.pvalue))
    print("dch vs score growth spearman correlation: {} {}".format(
        correlation_dch_score_growth.correlation, correlation_dch_score_growth.pvalue))
    print("dch vs simulated rewards spearman correlation: {} {}".format(
        correlation_dch_simulated_reward.correlation, correlation_dch_simulated_reward.pvalue))
    print("")
    print("ipw-dch vs post scores spearman correlation: {} {}".format(
        correlation_ipw_dch_post_score.correlation, correlation_ipw_dch_post_score.pvalue))
    print("ipw-dch vs traj length spearman correlation: {} {}".format(
        correlation_ipw_dch_traj_length.correlation, correlation_ipw_dch_traj_length.pvalue))
    print("ipw-dch vs score growth spearman correlation: {} {}".format(
        correlation_ipw_dch_score_growth.correlation, correlation_ipw_dch_score_growth.pvalue))
    print("ipw-dch vs simulated rewards spearman correlation: {} {}".format(
        correlation_ipw_dch_simulated_reward.correlation,
        correlation_ipw_dch_simulated_reward.pvalue))
    print("")
    print("{} {}".format(correlation_dch_post_score.correlation, correlation_dch_post_score.pvalue))
    print(
        "{} {}".format(correlation_dch_traj_length.correlation, correlation_dch_traj_length.pvalue))
    print("{} {}".format(correlation_dch_score_growth.correlation,
                         correlation_dch_score_growth.pvalue))
    print("{} {}".format(correlation_dch_simulated_reward.correlation,
                         correlation_dch_simulated_reward.pvalue))
    print("{} {}".format(correlation_ipw_dch_post_score.correlation,
                         correlation_ipw_dch_post_score.pvalue))
    print("{} {}".format(correlation_ipw_dch_traj_length.correlation,
                         correlation_ipw_dch_traj_length.pvalue))
    print("{} {}".format(correlation_ipw_dch_score_growth.correlation,
                         correlation_ipw_dch_score_growth.pvalue))
    print("{} {}".format(correlation_ipw_dch_simulated_reward.correlation,
                         correlation_ipw_dch_simulated_reward.pvalue))
    print("traj.-length  ", *traj_lengths)
    print("avg-dch  ", *np.round(avg_dch_list, 4))
    print("avg-ipw-dch  ", *np.round(avg_ipw_dch_list, 4))
    print("post-scores  ", *traj_posttest_score_list)
    print("score-growth  ", *np.round(traj_score_growth_list, 4))
    print("rewards  ", *np.round(plan_model.finalRewards, 4))
    return traj_lengths, avg_dch_list, avg_ipw_dch_list, traj_posttest_score_list, \
           traj_score_growth_list, plan_model.finalRewards, correlation_dch_post_score, \
           correlation_dch_traj_length, correlation_dch_score_growth, \
           correlation_dch_simulated_reward, correlation_ipw_dch_post_score, \
           correlation_ipw_dch_traj_length, correlation_ipw_dch_score_growth, \
           correlation_ipw_dch_simulated_reward


def generate_instructional_sequence(problem_topic_mapping):
    problem_seq = []
    topic_problem_mapping = {}
    for problem in problem_topic_mapping:
        topic = problem_topic_mapping[problem]
        if topic not in topic_problem_mapping:
            topic_problem_mapping[topic] = []
        topic_problem_mapping[topic].append(problem)

    topic_seq = ["Variables",
                 "Primitive_Data_Types",
                 "Constants",
                 "Arithmetic_Operations",
                 "Strings",
                 "Boolean_Expressions",
                 "Decisions",
                 "Switch",
                 "Exceptions",
                 "Loops_while",
                 "Loops_do_while",
                 "Loops_for",
                 "Nested_Loops",
                 "Objects",
                 "Classes",
                 "Arrays",
                 "Two-dimensional_Arrays",
                 "ArrayList",
                 "Inheritance",
                 "Interfaces",
                 "Wrapper_Classes"
                 ]
    for topic in topic_seq:
        problem_seq += sorted(topic_problem_mapping[topic])
    return problem_seq


if __name__ == '__main__':
    fold = 1
    top_k = 1  # top-k questions for recommendation

    # policy = "random"
    # policy = "mastery"
    # policy = "highest_prob_correct"
    # policy = "myopic"
    policy = "baseline"

    # reward_model = "logistic"
    reward_model = "linear"
    print("offline evaluation policy: {}, top_k: {}".format(policy, top_k))
    data_dir = "../data/MasteryGrids"

    fold_users_dict = pickle.load(open("../data/MasteryGrids/fold_users_dict.pkl", "rb"))
    problem_topic_mapping = pickle.load(
        open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))

    problem_seq = generate_instructional_sequence(problem_topic_mapping)

    user_name_pretest_posttest_scores = pickle.load(
        open("../data/MasteryGrids/user_name_pretest_posttest_scores.pkl", "rb"))
    user_records_dict = pickle.load(open("../data/MasteryGrids/user_records_dict.pkl", "rb"))
    next_questions_dict = pickle.load(open("{}/next_questions_dict.pkl".format(data_dir), "rb"))
    problem_name_problem_id_mapping = pickle.load(
        open("../data/MasteryGrids/problem_name_problem_id_mapping.pkl", "rb")
    )

    kcs = pickle.load(open("../data/MasteryGrids/kcs.pkl", "rb"))
    train_users_list = fold_users_dict[fold]["train"]
    test_users_list = fold_users_dict[fold]["test"]
    prBlue("train users: {} and test users: {}".format(len(train_users_list), len(test_users_list)))

    # use all training users' records to train the BKT model
    prBlue("loading trained BKT model")
    model_file_path = "../data/MasteryGrids/bkt_fold_{}_train_model.txt".format(fold)
    O, T, pi, kcMap = getYudelsonModel(model_file_path, len(kcs), kcs)
    # we need to make sure Shayan used same Yudelson Model as us
    # prior distribution: pi = [1-p(L_0), p(L_0)]
    # T is transition matrix:  is [[1-p(T), p(T)], [0, 1]]
    # O is observation matrix: is [[1-p(G), p(G)], [p(S), 1-p(S)]]
    # kcMap is topic name -> kc id

    train_bkt_X = []
    train_y = []
    init_belief = pi[:, 1]
    prBlue("train reward model based on pretest score and trained last belief from BKT")
    for index, user in enumerate(train_users_list):
        pretest, posttest = user_name_pretest_posttest_scores[user]
        print("{}: user: {}, pretest: {}, posttest: {}".format(
            index, user, pretest, posttest))
        # print("init belief: {}".format(list(init_belief)))
        # print("init mastery: {}".format(init_mastery))
        records = []
        for user, question, topic, result, _ in user_records_dict[user]:
            records.append([question, result])
        last_belief = compute_bkt_last_belief(pi, O, T, records, kcMap, problem_topic_mapping)

        q_list = []
        a_list = []
        for user, question, topic, result, _ in user_records_dict[user]:
            q_id = problem_name_problem_id_mapping[question]
            q_list.append(q_id)
            a_list.append(result)

        print("init belief: {}".format(list(init_belief)))
        print("last belief: {}".format(list(last_belief)))
        print("")
        x = np.append([pretest], last_belief)
        train_bkt_X.append(x)
        if reward_model == "logistic":
            train_y.append(round(posttest))
        elif reward_model == "linear":
            train_y.append(posttest)
        else:
            raise AttributeError
    prBlue("plan model reward model input shape: {}, output shape: {}".format(
        np.array(train_bkt_X).shape, np.array(train_y).shape))
    # reward model is logistic regression and the train_y should be binary
    # but if we binarize the train_y, it may become unbalanced data
    if reward_model == "logistic":
        planRewardModel = LogisticModel(train_bkt_X, train_y)
    elif reward_model == "linear":
        planRewardModel = LinearModel(train_bkt_X, train_y)
    else:
        raise AttributeError

    # initialize the domain model which is used to simulate student's response
    problem_list = list(problem_topic_mapping.keys())
    prBlue("number of problem: {} and they are: {}".format(len(problem_list), problem_list))
    plan_model = BKTModel(O, T, pi, kcMap, planRewardModel, problem_list, problem_topic_mapping,
                          top_k)

    lenTraj = []
    pretestList = []
    prBlue("loading testing users' info")
    users_q_list = []
    users_a_list = []
    users_name_list = []
    users_list = list(train_users_list) + list(test_users_list)
    for index, user in enumerate(users_list):
        users_name_list.append(user)
        traj_len = len(user_records_dict[user])
        q_list = []
        a_list = []
        for user, question, topic, result, _ in user_records_dict[user]:
            q_list.append(question)
            a_list.append(result)
        users_q_list.append(q_list)
        users_a_list.append(a_list)
        lenTraj.append(traj_len)
        pretest, posttest = user_name_pretest_posttest_scores[user]
        pretestList.append(pretest)
        print("{}: user: {}, traj len: {}, pretest: {}, posttest: {}".format(
            index, user, traj_len, pretest, posttest))
        print("question and answer pair: {}".format(list(zip(q_list, a_list))))
        print()
    if policy != "random":
        eval_policy(plan_model, policy, lenTraj, pretestList, users_q_list, users_a_list,
                    users_name_list, next_questions_dict, top_k)
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
            outputs = eval_policy(plan_model, policy, lenTraj, pretestList, users_q_list, users_a_list,
                                  users_name_list, next_questions_dict, top_k)
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
            plan_model.finalRewards = []

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
