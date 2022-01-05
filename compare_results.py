# -*- coding: utf-8 -*-
# @Author: jsgounot
# @Date:   2022-01-05 10:54:42
# @Last Modified by:   jsgounot
# @Last Modified time: 2022-01-05 10:54:51

import sys

f1, f2 = sys.argv[1], sys.argv[2]

def load_result(fname):
	r = {}
	with open(fname) as f:
		for idx, line in enumerate(f):
			if idx == 0:
				continue

			line = line.strip().split()
			name, res = line[0], tuple(line[1:])
			r[name] = res

	return r

r1 = load_result(f1)
r2 = load_result(f2)

# assert we have the same set of names
assert set(r1) == set(r2)

# assert we have the same results
for k1, v1 in r1.items():
	assert v1 == r2[k1]

print ('Same results')