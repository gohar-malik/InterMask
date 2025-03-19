import torch
import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_edge_list = [] 
for chain in t2m_kinematic_chain:
    for i, joint in enumerate(chain):
        if i == 0:
            continue
        t2m_edge_list.append([joint, chain[i-1]])
        t2m_edge_list.append([chain[i-1], joint])


t2m_edge_indices = torch.tensor(np.asarray(t2m_edge_list), dtype=torch.long).t().contiguous()

t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]


kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'

hhi_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0], # 9
                           [0,0,1],
                           [0,0,1],
                           [0,1,0], # 12
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1], # 15
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0], # 21
                           [0,1,0],
                           [0,0,1],
                           [0,0,1]
                           ])



hhi_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15, 22], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20], [15, 23], [15, 24]]
hhi_left_hand_chain = [[20, 37, 38, 39], [20, 25, 26, 27], [20, 28, 29, 30], [20, 34, 35, 36], [20, 31, 32, 33]]
hhi_right_hand_chain = [[21, 52, 53, 54], [21, 40, 41, 42], [21, 43, 44, 45], [21, 49, 50, 51], [21, 46, 47, 48]]
hhi_total_chain = hhi_kinematic_chain + hhi_left_hand_chain + hhi_right_hand_chain

hhi_edge_list = []
for chain in hhi_total_chain:
    for i, joint in enumerate(chain):
        if i == 0:
            continue
        hhi_edge_list.append([joint, chain[i-1]])
        hhi_edge_list.append([chain[i-1], joint])

hhi_edge_indices = torch.tensor(np.asarray(hhi_edge_list), dtype=torch.long).t().contiguous()



