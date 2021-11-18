import pandas as pd
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from offline_eval.utils import *
import seaborn as sns
import matplotlib.pyplot as plt


def process_raw_data():
    prRed("--- processing raw data ---")
    df = pd.read_csv("raw/IS172013Spring_2012Fall_2012Spring.csv")
    df = df[df["applabel"] == "QUIZJET"]
    df = df[["user", "group", "applabel", "parentname", "topicname", "result", "unixtimestamp"]]
    # sort records based on timestamp
    train_df = df.sort_values(['unixtimestamp'], ascending=True).reset_index(drop=True)

    count = 0
    user_records_dict = {}
    user_group_dict = {}
    parent_name_topic_name_dict = {}  # parentname is question
    topic_name_parent_name_dict = {}
    for user, group, applabel, parentname, topicname, result, unixtimestamp in train_df.to_numpy():
        count += 1
        if user not in user_group_dict:
            user_group_dict[user] = group
        if user not in user_records_dict:
            user_records_dict[user] = []
        user_records_dict[user].append([user, parentname, topicname, result, unixtimestamp])
        if parentname not in parent_name_topic_name_dict:
            parent_name_topic_name_dict[parentname] = topicname
        if topicname not in topic_name_parent_name_dict:
            topic_name_parent_name_dict[topicname] = {}
        if parentname not in topic_name_parent_name_dict[topicname]:
            topic_name_parent_name_dict[topicname][parentname] = 0
        topic_name_parent_name_dict[topicname][parentname] += 1
    seq_len_list = [len(user_records_dict[u]) for u in user_records_dict]
    prBlue("number of records: {}".format(count))
    prBlue("number of users: {}".format(len(user_records_dict)))
    prBlue("number of topics: {}".format(len(topic_name_parent_name_dict)))
    prBlue("number of questions: {}".format(len(parent_name_topic_name_dict)))
    prBlue("average user seq length: {:.2f} and std: {:.2f}".format(
        np.mean(seq_len_list), np.std(seq_len_list)))
    print("* save user records dict")
    pickle.dump(user_records_dict, open("user_records_dict.pkl", "wb"))

    print("generating next questions distribution")
    next_questions_dict = {}
    for user in list(user_records_dict.keys()):
        records = sorted(user_records_dict[user], key=lambda x: x[-1])
        user_records_dict[user] = records
        for index, (_, question, topic, score, timestamp) in enumerate(records[:-1]):
            if question not in next_questions_dict:
                next_questions_dict[question] = {}
            next_question = records[index + 1][1]
            if next_question not in next_questions_dict[question]:
                next_questions_dict[question][next_question] = 0
            next_questions_dict[question][next_question] += 1
    pickle.dump(next_questions_dict, open("next_questions_dict.pkl", "wb"))

    # mapping between question name and topic name
    kc_list = list(topic_name_parent_name_dict.keys())
    print("* save kc list ")
    pickle.dump(kc_list, open("kcs.pkl", "wb"))
    print("* save problem -> kc mapping ")
    pickle.dump(parent_name_topic_name_dict, open("problem_kc_mapping.pkl", "wb"))

    # process and get the mapping between user_name and [pretest, posttest] scores
    print("* process the user pre test and post test scores")
    print("print out pretest and posttest scores for duplicated user id")
    df = pd.read_csv("raw/pre_post_test.csv")
    df = df.dropna()
    user_id_test_scores = {}
    for user_id, pretest, posttest in df.to_numpy():
        user_id = int(user_id)
        if user_id not in user_id_test_scores:
            user_id_test_scores[user_id] = [pretest, posttest]
        else:
            print("duplicated user and test scores: {}, {}".format(user_id, pretest, posttest))
            print("exiting user test scores: {}, {}".format(user_id, user_id_test_scores[user_id]))
    prBlue("total number of users with test scores: {}".format(len(user_id_test_scores)))

    print("* normalize the test scores, and build user_name -> normalized test scores mapping")
    df = pd.read_csv("raw/studentMaps.csv")
    prBlue("all users test scores: [user_id, user_name, pretest, posttest]")
    user_name_test_scores = {}
    for user_name, user_id in df.to_numpy():
        if user_name not in user_group_dict or user_name not in user_records_dict:
            continue
        group = user_group_dict[user_name]
        if user_id in user_id_test_scores:
            pretest, posttest = user_id_test_scores[user_id]
            if group == "IS172013Spring":
                pretest = round(pretest / 16., 3)
                posttest = round(posttest / 18., 3)
            elif group == "IS172012Fall":
                pretest = round(pretest / 16., 3)
                posttest = round(posttest / 24., 3)
            elif group == "IS172012Spring":
                pretest = round(pretest / 20., 3)
                posttest = round(posttest / 20., 3)
            else:
                raise ValueError
            if pretest > 1. or posttest > 1.:
                raise ValueError
            user_name_test_scores[user_name] = [pretest, posttest]
            print(user_id, user_name, pretest, posttest)
    print("* save user_name -> test scores mapping")
    pickle.dump(user_name_test_scores, open("user_name_pretest_posttest_scores.pkl", "wb"))


def generate_bkt_data():
    """
    generate BKT data
    should use user_name_test_scores to generate train and test users

    :param user_records_dict:
    :param user_name_test_scores:
    :return:
    """
    prRed("--- generating data for BKT ---")
    user_name_test_scores = pickle.load(open("user_name_pretest_posttest_scores.pkl", "rb"))
    user_records_dict = pickle.load(open("user_records_dict.pkl", "rb"))

    print("* load user_name -> test scores mapping data, and generate k-folds splits")
    users = list(user_name_test_scores.keys())
    random.shuffle(users)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_users_dict = {}
    problem_name_problem_id_mapping = {}
    problem_id_problem_name_mapping = {}
    for fold, (train_index, test_index) in enumerate(kf.split(users)):
        print("fold {}".format(fold))
        print("train: {}, {}".format(len(train_index), np.array(users)[train_index]))
        print("test: {}, {}".format(len(test_index), np.array(users)[test_index]))

        train_users = np.array(users)[train_index]
        test_users = np.array(users)[test_index]
        fold_users_dict[fold + 1] = {"train": train_users, "test": test_users}

        print("save bkt training data and testing data as the input format of bkt model")
        train_f = open("bkt_input_fold_{}_train.csv".format(fold + 1), "w")
        test_f = open("bkt_input_fold_{}_test.csv".format(fold + 1), "w")
        for user in user_name_test_scores.keys():
            for user, parentname, topicname, result, unixtimestamp in user_records_dict[user]:
                if result == 0:
                    result = 2
                if user in train_users:
                    train_f.write("{}\t{}\t{}\t{}\n".format(result, user, parentname, topicname))
                elif user in test_users:
                    test_f.write("{}\t{}\t{}\t{}\n".format(result, user, parentname, topicname))

                if parentname not in problem_name_problem_id_mapping:
                    new_index = len(problem_name_problem_id_mapping) + 1
                    problem_name_problem_id_mapping[parentname] = new_index
                    problem_id_problem_name_mapping[new_index] = parentname
        train_f.close()
        test_f.close()
    print("* BKT model uses raw user_name, problem_name, and topic_name as input")
    print("* we should generate the problem_name and problem id for generating DKT data")
    print("* save fold users, and problem_name -> problem_id mapping, and vice versa")
    pickle.dump(fold_users_dict, open("fold_users_dict.pkl", "wb"))
    pickle.dump(problem_name_problem_id_mapping, open(
        "problem_name_problem_id_mapping.pkl", "wb"))
    pickle.dump(problem_id_problem_name_mapping, open(
        "problem_id_problem_name_mapping.pkl", "wb"))
    return fold_users_dict


def generate_data(user_records_dict, users, problem_name_problem_id_mapping):
    user_name_test_scores = pickle.load(open("user_name_pretest_posttest_scores.pkl", "rb"))
    data = {}
    q_data = []
    a_data = []
    q_dict = {}
    for user in users:
        pretest, posttest = user_name_test_scores[user]
        q_list = [0]
        a_list = [pretest]
        for user, parentname, topicname, result, unixtimestamp in user_records_dict[user]:
            problem_id = problem_name_problem_id_mapping[parentname]
            q_list.append(problem_id)
            a_list.append(result)
            if problem_id not in q_dict:
                q_dict[problem_id] = 0
            q_dict[problem_id] += 1
        q_data.append(q_list)
        a_data.append(a_list)
    data["q_data"] = q_data
    data["a_data"] = a_data
    data["num_users"] = len(users)
    data["num_questions"] = len(q_dict)
    data["num_records"] = np.sum(list(q_dict.values()))
    return data


def generate_dkt_data():
    prRed("--- generating data for DKT model ---")
    print("we should generate DKT data that is consistent with the BKT data")
    print("* loading fold_users_dict, users_records_dict, and problem_name -> problem id mapping")
    fold_users_dict = pickle.load(open("fold_users_dict.pkl", "rb"))
    user_records_dict = pickle.load(open("user_records_dict.pkl", "rb"))
    problem_name_problem_id_mapping = pickle.load(open(
        "problem_name_problem_id_mapping.pkl", "rb"))

    for fold in fold_users_dict.keys():
        fold_data = {}
        train_users = fold_users_dict[fold]["train"]
        test_users = fold_users_dict[fold]["test"]
        print("* generating and saving {} fold data, train users: {}, test users: {}".format(
            fold, len(train_users), len(test_users)))
        train_data = generate_data(user_records_dict, train_users, problem_name_problem_id_mapping)
        test_data = generate_data(user_records_dict, test_users, problem_name_problem_id_mapping)
        fold_data["train"] = train_data
        fold_data["test"] = test_data
        fold_data["num_users"] = len(train_users) + len(test_users)
        fold_data["num_questions"] = len(problem_name_problem_id_mapping)
        fold_data["num_records"] = train_data["num_records"] + test_data["num_records"]
        pickle.dump(fold_data, open("MasteryGrids_fold_{}.pkl".format(fold), "wb"))


def generate_figures():
    user_records_dict = pickle.load(open("user_records_dict.pkl", "rb"))
    user_name_test_scores = pickle.load(open("user_name_pretest_posttest_scores.pkl", "rb"))
    pretest_list = []
    posttest_list = []
    score_growth = []
    for user_name in user_name_test_scores:
        pretest, posttest = user_name_test_scores[user_name]
        pretest_list.append(pretest)
        posttest_list.append(posttest)
        score_growth.append(posttest - pretest)
    # sns.histplot(pretest_list, line_kws={"linewidth": 2}, color="red", kde=True)
    # sns.histplot(posttest_list, line_kws={"linewidth": 2}, color="green", kde=True)
    sns.histplot(score_growth, line_kws={"linewidth": 2}, color="blue", kde=True)
    # plt.xlabel("Student Pretest Score", fontsize=18)
    plt.xlabel("Knowledge Gain", fontsize=18)
    plt.ylabel("# of Students", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
    plt.savefig("knowledge_gain_distribution.pdf")
    plt.show()
    plt.clf()

    # print(len(user_name_test_scores))
    # count = 0
    # traj_list = []
    # for user_name in user_name_test_scores:
    #     print(user_name, len(user_records_dict[user_name]))
    #     size = len(user_records_dict[user_name])
    #     count += size
    #     traj_list.append(size)
    # print(count)
    # print(min(traj_list), np.median(traj_list), max(traj_list))


def generate_pear_fold_data(user_records_dict, users, pear_user_name_user_id_mapping,
                            pear_problem_name_problem_id_mapping):
    data = []
    for user_name in users:
        user_id = pear_user_name_user_id_mapping[user_name]
        records = []
        for user_name, parentname, topicname, result, unixtimestamp in sorted(
                user_records_dict[user_name], key=lambda x: x[-1]):
            time_index = len(records)
            problem_id = pear_problem_name_problem_id_mapping[parentname]
            records.append([user_id, time_index, problem_id, result])
        data.append(records)
    return data


def generate_pear_data():
    user_name_test_scores = pickle.load(open("user_name_pretest_posttest_scores.pkl", "rb"))
    user_id = 0
    pear_user_name_user_id_mapping = {}
    pear_user_id_user_name_mapping = {}
    for user_name in user_name_test_scores:
        if user_name not in pear_user_name_user_id_mapping:
            pear_user_name_user_id_mapping[user_name] = user_id
            pear_user_id_user_name_mapping[user_id] = user_name
            user_id += 1
    pickle.dump(pear_user_name_user_id_mapping, open("pear_user_name_user_id_mapping.pkl", "wb"))
    pickle.dump(pear_user_id_user_name_mapping, open("pear_user_id_user_name_mapping.pkl", "wb"))

    fold_users_dict = pickle.load(open("fold_users_dict.pkl", "rb"))
    user_records_dict = pickle.load(open("user_records_dict.pkl", "rb"))
    pear_problem_id_problem_name_mapping = {}
    pear_problem_name_problem_id_mapping = {}
    max_attempt = 0
    users_data = {}
    for user_name in user_name_test_scores:
        n_attempt = len(user_records_dict[user_name])
        if n_attempt > max_attempt:
            max_attempt = n_attempt
        user_id = pear_user_name_user_id_mapping[user_name]
        records = []
        for user_name, parentname, topicname, result, unixtimestamp in sorted(
                user_records_dict[user_name], key=lambda x: x[-1]):
            time_index = len(records)
            if parentname not in pear_problem_name_problem_id_mapping:
                problem_id = len(pear_problem_name_problem_id_mapping.keys())
                pear_problem_name_problem_id_mapping[parentname] = problem_id
                pear_problem_id_problem_name_mapping[problem_id] = parentname
            problem_id = pear_problem_name_problem_id_mapping[parentname]
            records.append([user_id, time_index, problem_id, result])
        users_data[user_id] = records
        pickle.dump(pear_problem_id_problem_name_mapping,
                    open("pear_problem_id_problem_name_mapping.pkl", "wb"))
        pickle.dump(pear_problem_name_problem_id_mapping,
                    open("pear_problem_name_problem_id_mapping.pkl", "wb"))

    for fold in fold_users_dict.keys():
        fold_data = {}
        train_users = fold_users_dict[fold]["train"]
        test_users = fold_users_dict[fold]["test"]
        print("* generating and saving {} fold data, train users: {}, test users: {}".format(
            fold, len(train_users), len(test_users)))

        train_data = generate_pear_fold_data(user_records_dict, train_users,
                                             pear_user_name_user_id_mapping,
                                             pear_problem_name_problem_id_mapping)
        test_data = generate_pear_fold_data(user_records_dict, test_users,
                                            pear_user_name_user_id_mapping,
                                            pear_problem_name_problem_id_mapping)
        test_user_list = []
        for user in test_users:
            user_id = pear_user_name_user_id_mapping[user]
            test_user_list.append(user_id)

        test_user_records = {}
        for records in test_data:
            for user_id, time_index, problem_id, result in records:
                if user_id not in test_user_records:
                    test_user_records[user_id] = {}
                if time_index not in test_user_records[user_id]:
                    test_user_records[user_id][time_index] = (problem_id, result)

        fold_data["train"] = train_data
        fold_data["test"] = test_data
        fold_data["num_users"] = len(train_users) + len(test_users)
        fold_data["num_questions"] = len(pear_problem_name_problem_id_mapping)
        fold_data["num_attempts"] = max_attempt
        fold_data["test_users"] = test_user_list
        fold_data["user_records"] = users_data
        fold_data["test_user_records"] = test_user_records
        pickle.dump(fold_data, open("MasteryGrids_pear_fold_{}.pkl".format(fold), "wb"))


if __name__ == '__main__':
    # process_raw_data()
    # generate_bkt_data()
    # generate_dkt_data()
    generate_pear_data()
    # generate_figures()
