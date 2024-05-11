# Train / Dev / Test split
# "manifest.json" -> "train_manifest.json", "dev_manifest.json", "eval_manifest.json"
# save path : "./manifest/"

import os
import argparse

train_size = 620000
#eval_size = 6000
manifest_path = "./manifest"

def _get_parser():
    """ Get arguments parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path', type=str,
                        default='./manifest')
    parser.add_argument('--train_size', type=int,
                        default=620000,
                        help='size of total dataset')
    # parser.add_argument('--eval_size', type=int,
    #                     default=6000)
    return parser

def main():
    parser = _get_parser()
    opt = parser.parse_args()
    
    with open(os.path.join(opt.manifest_path,"manifest.json"), "r") as f :
        lines = f.readlines()
        #print(len(lines))
        with open(os.path.join(opt.manifest_path,"train_manifest.json"), "w") as t :
            t.writelines(lines[:opt.train_size])
        with open(os.path.join(opt.manifest_path,"dev_manifest.json"), "w") as d :
            d.writelines(lines[opt.train_size:])

if __name__ == '__main__':
    main()