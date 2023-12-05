import os
import numpy as np
import regex as re

label_action =[
    {"id" : 0, "A001" : "drink water"},
    {"id" : 1, "A002" : "eating"},
    {"id" : 2, "A003" : "brushing teeth"},
    {"id" : 3, "A004" : "brushing hair"},
    {"id" : 4, "A005" : "drop"},
    {"id" : 5, "A006" : "pickup"},
    {"id" : 6, "A007" : "throw"},
    {"id" : 7, "A008" : "sitting down"},
    {"id" : 8, "A009" : "standing up"},
    {"id" : 9, "A010" : "clapping"},
    {"id" : 10, "A011" : "reading"},
    {"id" : 11, "A012" : "writing"},
    {"id" : 12, "A013" : "tear up paper"},
    {"id" : 13, "A014" : "wear jacket"},
    {"id" : 14, "A015" : "take off jacket"},
    {"id" : 15, "A016" : "wear a shoe"},
    {"id" : 16, "A017" : "take off a shoe"},
    {"id" : 17, "A018" : "wear on glasses"},
    {"id" : 18, "A019" : "take off glasses"},
    {"id" : 19, "A020" : "put on a hat/cap"},
    {"id" : 20, "A021" : "take off a hat/cap"},
    {"id" : 21, "A022" : "cheer up"},
    {"id" : 22, "A023" : "hand waving"},
    {"id" : 23, "A024" : "kicking something"},
    {"id" : 24, "A025" : "reach into pocket"},
    {"id" : 25, "A026" : "hopping (one foot jumping)"},
    {"id" : 26, "A027" : "jump up"},
    {"id" : 27, "A028" : "make a phone call/answer phone"},
    {"id" : 28, "A029" : "playing with phone/tablet"},
    {"id" : 29, "A030" : "typing on a keyboard"},
    {"id" : 30, "A031" : "pointing to something with finger"},
    {"id" : 31, "A032" : "taking a selfie"},
    {"id" : 32, "A033" : "check time (from watch)"},
    {"id" : 33, "A034" : "rub two hands together"},
    {"id" : 34, "A035" : "nod head/bow"},
    {"id" : 35, "A036" : "shake head"},
    {"id" : 36, "A037" : "wipe face"},
    {"id" : 37, "A038" : "salute"},
    {"id" : 38, "A039" : "put the palms together"},
    {"id" : 39, "A040" : "cross hands in front (say stop)"},
    {"id" : 40, "A041" : "sneeze/cough"},
    {"id" : 41, "A042" : "staggering"},
    {"id" : 42, "A043" : "falling"},
    {"id" : 43, "A044" : "touch head"},
    {"id" : 44, "A045" : "touch chest"},
    {"id" : 45, "A046" : "touch back"},
    {"id" : 46, "A047" : "touch neck"},
    {"id" : 47, "A048" : "nausea or vomiting condition"},
    {"id" : 48, "A049" : "feeling warm"}
]

file_name_regex = r"S(\d{3})C(\d{3})P(\d{3})R001A(\d{3})"
file_name_regex = re.compile(file_name_regex)

def get_label(file_name: str) -> int:
    print(file_name_regex.match(file_name) is None)
    label = file_name[-4:]
    for i in label_action:
        if label in i:
            return i["id"]
        
files = ["../../../dataset/Python/raw_npy/S010C001P008R002A044.skeleton.npy", "../../../dataset/Python/raw_npy/S010C002P008R002A044.skeleton.npy", "../../../dataset/Python/raw_npy/S010C003P008R002A044.skeleton.npy"]
for file in files:
    data = np.load(file, allow_pickle=True).item()
    print("Skeleton data shape: ", data["skel_body0"].shape)
    print(get_label(data["file_name"]))