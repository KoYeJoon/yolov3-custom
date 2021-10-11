# image name -> path name
import numpy as np
import os
import argparse

def parse_args():

    desc = "convert annotation image name to path name"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--file_path', type=str, default="annotation/train/train_labels.txt", help='file_path')
    parser.add_argument('--op',type=str, default="train",help="train/test mode")
    return parser.parse_args()


def change_image_to_path(file_path,op):
    new_file_path = "annotation/train/new_train_labels.txt"    
    if op=="test" :
        new_file_path = "annotation/test/new_test_labels.txt"

    with open(file_path, 'r') as file:
        reader = [ line.strip().split(' ') for line in file.readlines()]

    with open(new_file_path,"w") as txt_file:
            for line in reader:
                path = "data/images/train/"
                if op=="test":
                    path = "data/images/test/"
                line[0] = path+line[0]
                txt_file.write("%s "%line[0]+ " ".join([str(a) for a in line[1:]])+"\n")
                       


if __name__ == '__main__':
    args = parse_args()
    change_image_to_path(args.file_path,args.op)