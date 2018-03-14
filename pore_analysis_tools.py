import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import squareform, cdist
from sklearn.cluster import DBSCAN
from matplotlib import animation


# -------------------------------------------------------------------
# trajectory loading and slab selection
# ------------------------------------------------------------------

def load_respertime(infile):
    ''' Loads output of gmx mindist with commands -respertime and -or '''
    return np.loadtxt(infile, comments=('#', '@'))[:, 1:]


def select_z_slab(coords, z_from_mid):
    ''' Given a set of coordinates, calculates for each particle and time frame if the particle is within a certain
        distance from the bilayer midplane. Assumes that the bilayer is already centered on 0, and that a typical xyz
        coordinate array is the input

        Parameters:
            -coords - nframes * nparticles * 3dimensions
            -z_from_mid (nm) - will select in both directions
        Returns:
            -boolean array, size =first 2 dimensions of coords
    '''
    return np.abs(coords[:, :, 2]) <= z_from_mid


def assign_prot_chains(n_res_per_chain, n_chains):
    ''' Identifies chain of residues by an integer index starting from 0'''
    return np.array([ [i] * n_res_per_chain for i in range(n_chains) ]).flatten()


def z_bool_to_indices(sel):
    ''' Alternative tracking of waters in bilayer. Turns a boolean array into a list of arrays, where list[i] is itself
        a list of indices corresponding to the "i"th frame that were equal to 1 in the boolean array

        Parameters:
            -sel - nframes * nparticles boolean array
        Returns:
            -list - length nframes, with variable size of interal arrays
    '''
    return [np.where(sel[i, :])[0] for i in range(sel.shape[0]) ]


def calc_res_coms(traj):
    ''' returns center of mass xyz over time for all residues in trajectory '''
    outcoms = np.zeros((traj.n_frames, traj.n_residues, 3))
    for i in range(traj.n_residues):
        indices = traj.topology.select('resid {:d}'.format(i))
        outcoms[:, i, :] = traj.xyz[:, indices, :].mean(axis=1)
    return outcoms

# ----------------------------------------------------------------------------------------------------
# clustering functions
# ----------------------------------------------------------------------------------------------------


def mdtraj_distance_oneframe(traj, frame, sel):
    ''' The clustering algorithms we want to use can't natively account for periodic boundary conditions. So, what we
        will do is use the mdtraj distance calculation metrics (which account for PBC) to generate a squareform distance
        matrix of the particles we want to cluster, which will be fed into the clustering algorithms in lieu of
        the xyz parameters.

        This function operates on one frame at at time, but has the entire trajectory as input
    '''
    trajframe = traj.slice(frame, copy=False)
    particle_pairs = list(itertools.combinations(sel, 2))
    return squareform(md.compute_distances(trajframe, particle_pairs).flatten())


def cluster_one_frame(traj, sel, frame, method, *args, **kwargs):
    ''' calculates clusters on a single frame using desired method and parameters

    Parameters:
        -traj        - full mdtraj trajectory
        -sel         - atom selection (one frame)
        -frame       - simulation frame to slice
        -method      - clustering algorithm (ie, DBSCAN - must be imported)
        -args/kwargs - clustering paramters - for DBSCAN these would be eps and min_clust (or something like that)
    Returns:
        -labels      - n_particles array of labels. IF DBSCAN, -1 is an outlier and 0 to (n_clusters - 1) are labels
    '''
    dist_matrix = mdtraj_distance_oneframe(traj, frame, sel)
    return method(*args, **kwargs).fit(dist_matrix).labels_


def cluster_all_frames(traj, sel, method, *args, **kwargs):
    ''' Returns a list of lists of labels from cluster_one_frame '''
    return [ cluster_one_frame(traj, sel[i], i, method, *args, **kwargs) for i in range(traj.n_frames)]


def map_prot_to_pore(watercoords, watersel, pore_ids, protcoords, protsel):
    ''' Given selections of waters and proteins that qualify as pore forming, assign each protein residue to a pore.
        Will ignore outliers

    Parameters:
        -watercoords   - nframes * n_waters * xyz
        -watersel      - nframes list of lists of waters in pores
        -pore_ids      - size of watersel, pore identification number for each water
        -protcoords    - nframes * n_prot_res * xyz
        -protsel       - nframes list of lists of protein residues around pores

    Returns
        -prot_pore_ids - size of protsel, pore ids for each protein, corresponding to pore_ids numbering
    '''
    prot_pore_ids = []
    for i in range(watercoords.shape[0]):
        dists = cdist(watercoords[i, watersel[i], :], protcoords[i, protsel[i], :])
        dists[pore_ids[i] == -1, :] = 100
        prot_pore_ids.append(pore_ids[i][np.argmin(dists, axis=0)])
    return prot_pore_ids


def calc_cluster_info(label_list, max_clusts=15):
    ''' Calculates some statistics for the observed clusters over time. If not DBSCAN, likely the algorithm does not
        flag outliers.

    Parameters:
        -label_list - list of lists of labels from cluster_all_frames
        -max_clusts - for output array. Unused clusters are set to 0
    Returns
        -clust_sizes - n_frames * max_clusts array of number of particles per cluster, not including outliers
        -outliers    - number of outliers per frame
    '''
    outliers = np.zeros(len(label_list))
    clust_sizes = np.zeros((len(label_list), max_clusts))

    for i in range(len(label_list)):
        outliers[i] = np.sum(label_list[i] < 0)
        for j in range(max_clusts):
            clust_sizes[i, j] = np.sum(label_list[i] == j)
    return clust_sizes, outliers

# ----------------------------------------------------------------------------------------------------
# plotting functions
# ----------------------------------------------------------------------------------------------------


def scatter_scale_marker_by_z(coords, markerscale=[0, 20], zscale=1):
    ''' For making cooler plots. Takes in coordinates and scales the marker size by distance to the center of the
        bilayer. Markerscale gives [minimum, maximum] markersize, and zscale gives the maximum distance to linearly
        scale from 0 to maximum distance with sizes according to markerscale.

        Works with absolute value of z

        Works only on a frame by frame basis, designed to be callable inside animation update function.

        Parameters:
            coords - nparticles * xyz array
            markerscale - list of size 2
            zscale  - int (nm)

        Returns:
            markersizes - nparticles array
    '''
    return markerscale[1] - np.abs(coords[:, 2]) * (markerscale[1] - markerscale[0]) / zscale


def gen_cluster_colors_spectral(labels, max_clusters=5):
    ''' Using cluster_oneframe, you return a set of labels ranging from -1 (no cluster) to n_clusters - 1. This setup
        will allow us to set colors for clusters. No cluster is whatever label spectral=0 gives. Setting a fixed max
        number of clusters lets us keep the same color for cluster 0, 1 and so on up to max. The +1 in equation is to
        get everything into 0-positive space before normalizing to max labels

        Have to call on a frame-by-frame basis as we have different numbers of particles per frame
    '''
    return plt.cm.spectral((labels + 1) / (1 + max_clusters))


def animate_water_cluster_attempts(coords, sel, colors, interval=40):
    ''' Animates timecourse of just waters in z slab, colored by cluster

    Parameters:
        - coords   - n_frames * n_particles * xyz
        - sel      - n_frames list of indices to plot per frame
        - colors   - same dimensions as sel, with colors for each particle (should be based on cluster)
        - interval - animation speed
    '''

    anim_fig = plt.figure()
    txt = plt.text(3, 4, 'frame 0 / {:d}'.format(coords.shape[0]))
    plt.xlim(coords[:, :, 0].min(), coords[:, :, 0].max())
    plt.ylim(coords[:, :, 1].min(), coords[:, :, 1].max())

    waterdata = plt.scatter([], [])

    def anim_update(i):
        txt.set_text('frame {:d} / {:d}'.format(i, coords.shape[0]))
        waterdata.set_offsets(coords[i, sel[i], 0:2])
        waterdata.set_color(colors[i])
        waterdata.set_sizes(scatter_scale_marker_by_z(coords[i, sel[i], :], markerscale=[0, 50]))
    ani = animation.FuncAnimation(anim_fig, anim_update, frames=coords.shape[0], interval=interval)
    plt.show()


def animate_prot_cluster_mapping(watercoords, watersel, watercolors, protcoords, protsel, protcolors, interval=40):
    ''' Animates timecourse of waters and proteins in z slab, colored by cluster assignment

    Parameters:
        - watercoords   - n_frames * n_particles * xyz
        - watersel      - n_frames list of indices of watersto plot per frame
        - watercolors   - same dimensions as watersel, with colors for each particle (should be based on cluster)
        - protcoords    - n_frames * n_res * xyz
        - protsel       - n_Frames list of indices of residue COMS per frame
        - protcolors    - same dimensions as protsel
        - interval - animation speed
    '''

    anim_fig = plt.figure()
    txt = plt.text(3, 4, 'frame 0 / {:d}'.format(watercoords.shape[0]))
    plt.xlim(watercoords[:, :, 0].min(), watercoords[:, :, 0].max())
    plt.ylim(watercoords[:, :, 1].min(), watercoords[:, :, 1].max())

    waterdata = plt.scatter([], [])
    protdata  = plt.scatter([], [], marker='^')

    def anim_update(i):
        txt.set_text('frame {:d} / {:d}'.format(i, watercoords.shape[0]))
        waterdata.set_offsets(watercoords[i, watersel[i], 0:2])
        waterdata.set_color(watercolors[i])
        waterdata.set_sizes(scatter_scale_marker_by_z(watercoords[i, watersel[i], :], markerscale=[0, 50]))

        protdata.set_offsets(protcoords[i, protsel[i], 0:2])
        protdata.set_color(protcolors[i])
        protdata.set_sizes(scatter_scale_marker_by_z(protcoords[i, protsel[i], :], markerscale=[0, 50]))

    ani = animation.FuncAnimation(anim_fig, anim_update, frames=watercoords.shape[0], interval=interval)
    plt.show()


def animate_water_in_pores(watercoords, watersel, protcoords, protsel, protchains, watermarkermax=30, protmarkermax=50):
    ''' Animates timecourse of waters and proteins in a z slab. The waters are blue, the proteins are a distribution
        of colors according to gen_cluster_colors_spectral, and are colored by chain.

        Sizes of particles are dynamic, the closer they are to the center, the larger they are, colored by
        scatter_scale_marker_by_z, and upper bounded by watermarkermax and protmarkermax

        Parameters:
            watercoords - n_frames * n_waters * xyz
            watersel    - n_frames list of list of water indices
            protcoords  - n_frames * n_residues * xyz
            protsel     - n_frames list of list of residue indices
            protchains  - n_frames * n_residues array of chain assignments
    '''
    anim_fig = plt.figure()

    def anim_update(i):

        plt.clf()
        plt.xlim(watercoords[:, :, 0].min(), watercoords[:, :, 0].max())
        plt.ylim(watercoords[:, :, 1].min(), watercoords[:, :, 1].max())
        plt.text(3, 4, 'frame {:d}/{:d}'.format(i, watercoords.shape[0]))
        plt.scatter(watercoords[i, watersel[i], 0], watercoords[i, watersel[i], 1],
                    s=scatter_scale_marker_by_z(watercoords[i, watersel[i], :], markerscale=[0, watermarkermax]), c='b')
        plt.scatter(protcoords[i, protsel[i], 0], protcoords[i, protsel[i], 1],
                    s=scatter_scale_marker_by_z(protcoords[i, protsel[i], :], markerscale=[0, protmarkermax]),
                    c=plt.cm.spectral(protchains[protsel[i]] / protchains[-1]), marker='v')

    ani = animation.FuncAnimation(anim_fig, anim_update, frames=watercoords.shape[0], interval=20)
    plt.show()

print(__name__)

if __name__ == "__main__":

    # cutoff parameters and stuff, can change
    water_contact_cutoff = 0.3
    n_res_per_chain = 44
    n_chains =  11
    z_slab_updown = 0.5

    # water loading and z selection
    watergro = 'water_heavy.gro'
    watertrr = 'water_heavy.trr'
    watertraj = md.load_trr(watertrr, top=watergro)
    water_slab_bool = select_z_slab(watertraj.xyz, z_slab_updown)
    water_slab_list = z_bool_to_indices(water_slab_bool)

    # protein loading and z selection - on a residue COM basis, not atomic
    protgro = 'prot_heavy.gro'
    prottrr = 'prot_heavy.trr'
    prottraj = md.load_trr(prottrr, top=protgro)
    prot_rescoms = calc_res_coms(prottraj)
    prot_slab_bool = select_z_slab(prot_rescoms, z_slab_updown)
    prot_slab_list = z_bool_to_indices(prot_slab_bool)

    # protein filtering based on water contacts
    prot_water_mindist = load_respertime('mindist_by_res.xvg')
    prot_water_contact_bool = prot_water_mindist < water_contact_cutoff
    prot_chain_indices = assign_prot_chains(n_res_per_chain, n_chains)
    prot_composite_keep = prot_water_contact_bool & prot_slab_bool
    prot_composite_list  = z_bool_to_indices(prot_composite_keep)  # both z and water contact based filter selection

    # demo animation with proteins
    animate_water_in_pores(watertraj.xyz, water_slab_list, prot_rescoms, prot_composite_list, prot_chain_indices)

    # demo clustering
    water_clusters = cluster_all_frames(watertraj, water_slab_list, DBSCAN,
                                        metric='precomputed', eps=1.0, min_samples=5)
    water_colors = [ gen_cluster_colors_spectral(water_clusters[i], 7) for i in range(len(water_clusters)) ]
    animate_water_cluster_attempts(watertraj.xyz, water_slab_list, water_colors)

    # assigning proteins to pores
    prot_clusters = map_prot_to_pore(watertraj.xyz, water_slab_list, water_clusters,  prot_rescoms, prot_composite_list)
    prot_colors =  [gen_cluster_colors_spectral(prot_clusters[i], 7) for i in range(len(prot_clusters)) ]
    animate_prot_cluster_mapping(watertraj.xyz, water_slab_list, water_colors,
                                 prot_rescoms, prot_composite_list, prot_colors)

    # statistics on clustering
    cluster_sizes, outliers = calc_cluster_info(water_clusters)
    cluster_sizes.sort(axis=1)    # may not be ideal for some analyses, keeps largest at index 0 and so on per frame
    n_clusters = np.sum(cluster_sizes > 0, axis=1)
