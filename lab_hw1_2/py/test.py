#!/usr/bin/env python3

import sys, os.path
os.chdir(os.path.dirname(sys.argv[0]))

def parse_list(s):
    code = f"[{s[s.index('{') + 1 : s.rindex('}')]}]".replace('\n', '')
    print(code)
    return eval(code)

with open("../data/test_set.h") as fts, \
        open("../data/label.h") as flb:
    lts = parse_list(fts.read())
    llb = parse_list(flb.read())
