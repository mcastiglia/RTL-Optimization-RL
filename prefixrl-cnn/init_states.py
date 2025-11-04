import numpy as np
import copy
import global_vars
from environment import Graph_State

# Initialize the graph based on the type of adder and return the initial state
def init_graph(n: int, type: int):
    if type == 0:
        return init_graph_serial(n)
    elif type == 1:
        return init_graph_sklansky(n)
    elif type == 2:
        return init_graph_brent_kung(n)
    else:
        raise ValueError(f"Invalid initial adder type: {type}")
    
# Algorithm 1 (Initialize) from PrefixRL
def init_graph_serial(n: int):
    nodelist = np.zeros((n, n))
    levellist = np.zeros((n, n))
    minlist = np.zeros((n, n))
    
    for m in range(n):
        nodelist[m, m] = 1
        nodelist[m, 0] = 1
        levellist[m, m] = 1
        levellist[m, 0] = m+1
    level = levellist.max()
    minlist = copy.deepcopy(nodelist)
    for m in range(n):
        minlist[m, m] = 0
        minlist[m, 0] = 0
    size = nodelist.sum() - n
    
    state = Graph_State(level, n, size, nodelist, levellist, minlist, 0)
    state.update_fanoutlist()
    return state
    
# Initialize sklansky graph (Taken from ArithTreeRL)
def init_graph_sklansky(n: int):
    nodelist = np.zeros((n, n))
    levellist = np.zeros((n, n))
    for m in range(n):
        nodelist[m, m] = 1
        levellist[m, m] = 1
        t = m
        now = m
        x = 1
        level = 1
        while t > 0:
            if t % 2 ==1:
                last_now = now
                now -= x
                nodelist[m, now] = 1
                levellist[m, now] = max(level, levellist[last_now-1, now]) + 1
                level += 1
            t = t // 2
            x *= 2
    
    minlist = copy.deepcopy(nodelist)
    for m in range(n):
        minlist[m, m] = 0
        minlist[m, 0] = 0
    
    level = levellist.max()
    size = nodelist.sum() - n
    
    state = Graph_State(level, n, size, nodelist, levellist, minlist, 0)
    state.nodelist, state.minlist = state.legalize(nodelist, minlist)
    state.update_fanoutlist()
    return state

# Initial Brent-Kung graph (Taken from ArithTreeRL)
def init_graph_brent_kung(n: int):
    def update_level_map(nodelist, levellist):
        levellist[1:].fill(0)
        levellist[0, 0] = 1
        for m in range(1, n):
            levellist[m, m] = 1
            prev_l = m
            for l in range(m-1, -1, -1):
                if nodelist[m, l] == 1:
                    levellist[m, l] = max(levellist[m, prev_l], levellist[prev_l-1, l])+ 1
                    prev_l = l
        return levellist

    nodelist = np.zeros((n, n))
    levellist = np.zeros((n, n))
    for i in range(n):
        nodelist[i, i] = 1 
        nodelist[i, 0] = 1
    t = 2
    while t < n:
        for i in range(t-1, n, t):
            nodelist[i, i-t+1] = 1
        t *= 2
    levellist = update_level_map(nodelist, levellist)
    level = levellist.max()
    minlist = copy.deepcopy(nodelist)
    for i in range(n):
        minlist[i, i] = 0
        minlist[i, 0] = 0
    size = nodelist.sum() - n
    print("BK level ={}, size = {}".format(levellist.max(), nodelist.sum()-n))
    state = Graph_State(level, n, size, nodelist, levellist, minlist, 0)
    state.update_fanoutlist()
    return state
        
        