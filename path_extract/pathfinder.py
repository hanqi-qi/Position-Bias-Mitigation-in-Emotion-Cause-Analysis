"""Extract paths from ConceptNet, based on code at https://github.com/INK-USC/KagNet"""
import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import numpy as np
import csv


config = configparser.ConfigParser()
config.read("../paths.cfg")


cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None




def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")

def load_cpnet():
    global cpnet,concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def get_edge(src_concept, tgt_concept):
    global cpnet, concept2id, relation2id, id2relation, id2concept
    rel_list = cpnet[src_concept][tgt_concept]
    # tmp = [rel_list[item]["weight"] for item in rel_list]
    # s = tmp.index(min(tmp))
    # rel = rel_list[s]["rel"]
    return list(set([rel_list[item]["rel"] for item in rel_list]))

# source and target is text
def find_paths(source, target, ifprint = False):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    #if no match word, just skip the keyword:
    if (not source in concept2id.keys()) or (not target in concept2id.keys()):
        print("No avaiable Emotion word or keywords in concept dict")
        return []
    else:
        s = concept2id[source]
        t = concept2id[target]

    # try:
    #     lenth, path = nx.bidirectional_dijkstra(cpnet, source=s, target=t, weight="weight")
    #     # print(lenth)
    #     print(path)
    # except nx.NetworkXNoPath:
    #     print("no path")
    # paths = [path]

        if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
            return
        all_path = []
        all_path_set = set()

        for max_len in range(1, 4):
            for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=max_len):
                path_str = "-".join([str(c) for c in p])
                if path_str not in all_path_set:
                    all_path_set.add(path_str)
                    all_path.append(p)
                if len(all_path) >= 100:  # top shortest 300 paths
                    break
            if len(all_path) >= 100:  # top shortest 300 paths
                break
        if len(all_path)<1:
            print("keyword_pairs:0: {Not relevant to the emotion")
            print("}")
            return []
        all_path.sort(key=len, reverse=False)
        pf_res = []
        for p in all_path:
            # print([id2concept[i] for i in p])
            rl = []
            for src in range(len(p) - 1):
                src_concept = p[src]
                tgt_concept = p[src + 1]

                rel_list = get_edge(src_concept, tgt_concept)
                rl.append(rel_list)
                if ifprint:
                    rel_list_str = []
                    for rel in rel_list:
                        if rel < len(id2relation):
                            rel_list_str.append(id2relation[rel])
                        else:
                            rel_list_str.append(id2relation[rel - len(id2relation)]+"*")
                    print(id2concept[src_concept], "----[%s]---> " %("/".join(rel_list_str)), end="")
                    if src + 1 == len(p) - 1:
                        print(id2concept[tgt_concept], end="")
            if ifprint:
                print()
        return pf_res

#TODO
def cause_filter(rl): #less than 4-hops
    flag = True
    rl = np.array(rl)
    if (np.any(rl==0)) or (np.any(rl==12)) or (np.any(rl==13)) or rl.shape[0]>4:
        flag = False
    return flag


def process(filename, batch_id=-1):
    pf = []
    output_path = filename + ".%d" % (batch_id) + ".pf"
    import os
    if os.path.exists(output_path):
        print(output_path + " exists. Skip!")
        return

    load_resources()
    load_cpnet()
    with open(filename, 'r') as fp:
        mcp_data = json.load(fp)
        mcp_data = list(np.array_split(mcp_data, 100)[batch_id])

        for item in tqdm(mcp_data, desc="batch_id: %d "%batch_id):
            acs = item["ac"]
            qcs = item["qc"]
            pfr_qa = []  # path finding results
            for ac in acs:
                for qc in qcs:
                    pf_res = find_paths(qc, ac)
                    pfr_qa.append({"ac":ac, "qc":qc, "pf_res":pf_res})
            pf.append(pfr_qa)

    with open(output_path, 'w') as fi:
        json.dump(pf, fi)


if __name__ == '__main__':
    load_resources()
    load_cpnet()
    #>>>>>>>> show paths for case study >>>>>>>##
    # find_paths("survival", "touch", ifprint=True)
    # print("--------")
    # print();print();print();print();print();

    sys.stdout = open('raw_paths.log', 'w')
    keywords_file = open('/input_file/keywords_en.csv').readlines()#extracted keywords in english
    ori_file = open("/input_file/clause_keywords.txt").readlines()

    ch_words = open("/words.txt").readlines()
    en_words = open("world_translation.txt").readlines()
    ch_en_dict = {}
    for i in range(len(ch_words)):
        ch_en_dict[ch_words[i].strip()] = en_words[i].strip()


    whole_data =[]
    doc_id = 1

    print("document%d: "%doc_id)
    for idx in range(len(ori_file)):
        next_id = int(ori_file[idx].split(",")[0])
        if next_id > doc_id and next_id>1:
            doc_id = next_id
            print("document%d:"%doc_id)
        print("sen_%d: "%int(ori_file[idx].split(",")[1]))
        print(idx)

        emotion_word = (ori_file[idx].split(",")[-1]).strip()
        keywords = keywords_file[idx].split(",")
        if len(keywords_file[idx])<2: #no keywords
            print("keyword_pairs:0: {Not relevant to the emotion")
            print("}")
        else:
            for kw_id in range(len(keywords)):
                if (keywords[kw_id].strip()).lower() == emotion_word.lower():
                    print("keyword_pairs:0: {%s is the emotion expression"%emotion_word)
                    print("}")
                else:
                    print("keyword_pairs:%d: {"%kw_id)
                    find_paths((keywords[kw_id].strip()).lower(), emotion_word, ifprint=True)
                    print("}")

"""Test code"""
# keywords_file = open('../path_code/input_file/keywords_en.csv').readlines()
# ori_file = open("../path_code/input_file/clause_keywords.csv").readlines()

# emo_dict = {}
# doc_id = 0 
# doc_clause = {}
# Num_clause_path = []
# for i in range(len(ori_file)):
#     keywords = keywords_file[i].split(",") #keywords list
#     #padding paths for each clause
#     clause_paths = []
#     for kw in keywords:
#         pf_res = find_paths(kw, ori_file[i][2], ifprint=True)
#         clause_paths.append(pf_res) # calculate_path collection, should be easy to parse
#     Num_clause_path.append(clause_paths)


