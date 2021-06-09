'''Filter the raw extracted path from ConceptNet
input_files: path_data/wordCh/wordEn'''

import statistics as sta
import collections
import re
from nltk.stem.snowball import SnowballStemmer
import json


stemmer = SnowballStemmer("english")

path_data = open('').readlines()

new_data = []
path_num_list=[]
path_data_list = []
read_path = False
path_num =0
sen_num,doc_num = 0,0

ch_words = open("./data/wordCh.txt").readlines()#wordCh
en_words = open("./data/wordEn.txt").readlines()#wordEn
en2ch_dict = {}
for i in range(len(ch_words)):
    en2ch_dict[en_words[i].strip()] = ch_words[i].strip()



for line in path_data:
    if len(re.findall('sen_[0-9]+',line))>0 and read_path == True:
        path_num_list.append(path_num)
        # print(path_num)
        path_num =0
        path_data_list.append(new_data)
        new_data = []
        sen_num = sen_num+1
        # new_data.append(line)
    elif '->' in line:
        path = line.strip()
        read_path = True

        if ('notdesires' in path) or ("antonym" in path) or ("notcapableof" in path) or ("/" in path) or ("[]" in path):
            # print("noisy path")
            continue
        else:
            new_data.append(line.strip())
            path_num = path_num+1
    elif 'document' in line:
        doc_num = doc_num +1
    else:
        continue
    #at the end of the doc
path_data_list.append(new_data)
path_num_list.append(path_num)

print(max(path_num_list),min(path_num_list),sta.mean(path_num_list),sta.median(path_num_list))



print(len(path_data_list))
print(len(path_num_list))

print("Check out the path_data and the its content")
#assert the len(path_data_list)==31296
print(len(path_data_list[-1]))
print(path_num_list[-1])

#keep only ten paths
counter=collections.Counter(path_num_list)
print(counter)
print(sum(path_num_list))

#according to the counter, make sure the hyper-parameter of the maximum number of paths kept
max_path_num = 15
'''calculate the top10 paths based on tf-idf value and relation types, otherwise padding the paths until reaching the set number'''

Ren2ch_dict = {
    "isa": "是",
    "relatedto": "相关",
    "hassubevent": "然后",
    "causes": "导致",
    "capableof": "能够",
    "desires": "想要",
    "atlocation": "在",
    "usedfor": "用来",
    "partof":"组成",
    "hasa":"有",
    "hasproperty":"有",
    "motivatedbygoal":"想要",
    "createdby":"相关",
    "synonym":"是"
}


def return_top10_paths(paths):
    # result_en_path = paths[:10]
    ch_path_list = []
    for en_path in paths:
        ch_path = en2ch_path(en_path)
        if len(ch_path)>0:
            ch_path_list.append(ch_path)
        else:
            continue

    if len(ch_path_list)>14 :
        return ch_path_list[:15]
    else:
        for res_id in range(15-len(ch_path_list)):
            ch_path_list.append(["0"])
    return ch_path_list

def en2ch_path(en_path):
    idx = 0
    path = []
    for item in en_path.split(" "):
        if idx%2==0: #entity
            en_entity=stemmer.stem(item)
            if en_entity in en2ch_dict.keys():
                ch_entity = en2ch_dict[en_entity]
                path.append(ch_entity)
            else:
                return []
        else:#relation
            relation = "".join(re.findall(r'[A-Za-z]', item))
            if relation in Ren2ch_dict.keys():
                path.append(Ren2ch_dict[relation])
            else:
                return []
        idx +=1
    return path

path_file = open("path_data_1120_num15.txt",'w')

filtered_paths = []
for paths in path_data_list:
    filtered_paths.append(return_top10_paths(paths))
    for line in return_top10_paths(paths):
        path_file.writelines(" ".join(line)+'\n')

print(len(filtered_paths),filtered_paths[0])
path_file.close()




