import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R
from utils import paramUtil
from visualization.joints2bvh import Joint2BVHConvertor
from visualization.remove_fs import *
from os.path import join as pjoin
# import cv2

bvh_converter = Joint2BVHConvertor()

def preprocess_plot_motion(motions, caption, vis_dir, npy_dir, file_name, foot_ik=False):
    """
    motions: (seq_len, 2, 262)
    """
    sequences = []
    sequences_ik = []
    foots = []
    for j in range(2):
        motion = motions[:,j]
        joints3d = motion[:,:22*3].reshape(-1,22,3)
        joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
        sequences.append(joints3d)

        rot6d = motion[:,22*3*2:-4].reshape(-1,21,6)
        rot6d = np.concatenate([np.zeros((rot6d.shape[0], 1, 6)), rot6d], axis=1)
        
        np.save(pjoin(npy_dir, file_name + f"_{j}.npy"), np.concatenate([joints3d, rot6d], axis=-1))
        
        if foot_ik:
            ik_joint, foot = remove_fs(joints3d, None, fid_l=(7, 10), fid_r=(8, 11), interp_length=5,
                                  force_on_floor=True)
            ik_joint = filters.gaussian_filter1d(ik_joint, 1, axis=0, mode='nearest')
            sequences_ik.append(ik_joint) 
            foots.append(foot)

            np.save(pjoin(npy_dir, "ik_" + file_name + f"_{j}.npy"), np.concatenate([ik_joint, rot6d], axis=-1))
            
            
            

    plot_3d_motion_2views(pjoin(vis_dir, file_name + f".mp4"), paramUtil.t2m_kinematic_chain, sequences, title=caption, fps=30)
    if foot_ik:
        plot_3d_motion_2views(pjoin(vis_dir, "ik_" + file_name + f".mp4"), paramUtil.t2m_kinematic_chain, sequences_ik, title=caption, fps=30, foots=None)


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    # print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)


        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        # data[:, :, 0] -= data[0:1, 0:1, 0]
        # data[:, :, 0] += mp_offset[i]
        #
        # data[:, :, 2] -= data[0:1, 0:1, 2]
        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 15#7.5
        #         ax =
        plot_xzPlane(-3, 3, 0, -3, 3)
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()
    
def plot_3d_motion_2views(save_path, kinematic_tree, mp_joints, title, figsize=(20, 10), fps=120, radius=8, foots=None):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 4])
        ax.set_zlim3d([0, radius / 4])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    fig = plt.figure(figsize=figsize)
    axs = []
    axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
    axs.append(fig.add_subplot(1, 2, 2, projection='3d'))
    for ax in axs: init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    colors = ['orange', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    rot = R.from_euler('y', 110, degrees=True)
    for i,joints in enumerate(mp_joints):

        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)



        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        
        data_rot = rot.apply(data.reshape(-1,3)).reshape(-1,22,3)
    
        mp_data.append({"joints":data,
                        "joints_rot":data_rot,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })


    def update(index):
        for ax in axs:
            ax.lines = []
            ax.collections = []

            ax.dist = 15
            plot_xzPlane(ax, -3, 3, 0, -3, 3)
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=120, azim=270)
            ax.axis('off')
        
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                if foots is not None:
                    l_heel, l_toe, r_heel, r_toe =  foots[pid][:,index]
                    if l_toe == 1:
                        color = 'darkred'
                axs[0].plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
                axs[1].plot3D(data["joints_rot"][index, chain, 0], data["joints_rot"][index, chain, 1], data["joints_rot"][index, chain, 2], linewidth=linewidth,
                          color=color)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()
