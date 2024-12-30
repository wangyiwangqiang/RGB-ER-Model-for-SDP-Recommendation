# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:29:25 2023

@author: Janus_yu
"""

import heapq
a_list = [3, 4, 2, 5, 1, 6]
c_dict = {'A':[2,3], 'B':[1,500], 'C':[6]}
topNum = 2

K_max_item_score = heapq.nlargest(3, c_dict, key=c_dict.get)
print(K_max_item_score)
