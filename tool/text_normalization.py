import re
import sys

def process(src_str):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    return re.sub(r"[{0}]+".format(punc), "", src_str).upper()

if __name__ == "__main__":
    for line in sys.stdin.readlines():
        utt, text = line.strip().split(maxsplit=1)
        print(utt + ' ' + process(text))
