import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, scale

from copkmeans.cop_kmeans import cop_kmeans
import time

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import SpectralClustering
import sys

from ddCRP import ddCRP
from ddCRP import Priors


import operator

run_common_frames = False
run_distance_matrix = False

def distance(input, ids, alpha_pc, cannot_link,  i, j):
    if i==j:
        return 0
    intra_similarities = cosine_similarity(input[input[:, 1] == ids[i]][:, 10:],
                                           input[input[:, 1] == ids[j]][:, 10:]).reshape(-1)
    if (i,j) in cannot_link or (j,i) in cannot_link:
        return 1
    return (1 - alpha_pc * np.mean(intra_similarities) - (1 - alpha_pc) * np.max(
        intra_similarities)) / 2


def run(input_file, output_file, max_common_frames, n_clusters, alpha_pc=1, version=2):
    input = np.load(input_file)
    #Frame_id, track id, ., ., ., ., ., ., ., ., appearance features

    t_ = time.time()

    print("Input shape:", input.shape)
    ids = list(np.unique(input[:,1]))

    ######## NETTOYAGE

    print("Total number of ids:",len(ids))
    ids_by_frames = {}
    for row in input:
        if row[0] not in ids_by_frames.keys():
            ids_by_frames[row[0]] = []
        ids_by_frames[row[0]].append(row[1])
    n_ids_by_frames = {k: len(v) for k, v in ids_by_frames.items()}


    #plt.bar(n_ids_by_frames.keys(), n_ids_by_frames.values(), color='g')
    #plt.show()


    print("Maximum number of ids on the same frame:", max(n_ids_by_frames.values()))
    ff = []
    for f, nid in n_ids_by_frames.items():
        if nid > n_clusters:
            ff.append(f)
    print("Delete frames with n_detections > n_clusters:",len(ff),ff)
    input = np.array([x for x in input if x[0] not in ff])
    min_len_tracklet = 10

    lens = []
    to_remove=[]
    for i in ids:
        t = input[input[:, 1] == i][:, 0]
        if t.shape[0] < min_len_tracklet:
            to_remove.append(i)
        else:
            lens.append(t.shape[0])
    for i in to_remove:
        ids.remove(i)
        input = input[~(input[:, 1] == i)]
    print("Delete tracklets with n_detections < ", min_len_tracklet, " : ", len(to_remove))
    print(input.shape,"detections x features")
    print("Mean len of tracklets (in frames):", np.mean(lens),)

    ######## FIN NETTOYAGE



    random_data = []
    data = []

    nn_frames = []
    for i in ids:
        group = input[:, 1] == i
        n_frames = input[group].shape[0]
        nn_frames.append(n_frames)
        d = np.zeros(input[0,10:].shape[0]+2)
        d[0] = input[group][:, 0].min(axis=0)
        d[1] = i
        d[2:] = input[group][:, 10:].mean(axis=0)
        #d[2:] = d[2:] / np.linalg.norm(d[2:])
        data.append(d)

        x = np.random.random(128)
        x = x / np.linalg.norm(x)
        random_data.append(list(d[:2]) + list(x))

    data = np.array(data)
    data = data[data[:, 0].argsort()]#Sort by asc frame_idx
    ids = list(data[:,1])

    plt.hist(nn_frames, bins=100)
    #plt.show()

    common_frames = np.zeros((len(ids),len(ids)))
    if run_common_frames:
        print("Computing common frames matrix...")
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                n_common_frames = len(set(list(input[input[:, 1] == ids[i]][:, 0])).intersection(list(input[input[:, 1] == ids[j]][:, 0])))
                common_frames[i,j] = n_common_frames
        print("Saved common frames matrix")
        np.save("common_frames.npy", common_frames)
    else:
        print("Loaded common frames matrix")
        common_frames = np.load("common_frames.npy")

    print("Computing 'cannot link' constraints with max_common_frames = ",max_common_frames)
    must_link = []
    cannot_link = [(i,j) for i in range(len(ids)) for j in range(i+1, len(ids)) if common_frames[i,j] > max_common_frames]
    can_link = [[i,j] for i in range(len(ids)) for j in range(i+1, len(ids)) if common_frames[i,j] <= max_common_frames]

    print("Number of constraints", len(cannot_link))


    if version == 2: #SPECTRAL CLUSTERING
        if run_distance_matrix:
            print("Computing distance matrix...")

            distance_matrix = np.zeros((len(ids), len(ids)))
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    distance_matrix[i, j] = distance(input, ids, alpha_pc, cannot_link,  i, j)

                    if (i,j) in cannot_link:
                        distance_matrix[i, j] = 1

                    distance_matrix[j, i] = distance_matrix[i, j]

            print("Saved distance matrix")
            np.save("distance_matrix.npy", distance_matrix)
        else:
            print("Loaded distance matrix")
            distance_matrix = np.load("distance_matrix.npy")

        np.set_printoptions(threshold=sys.maxsize)


        from ddCRP import synthetic

        X = scale(data[:,2:])

        alpha = 10
        mu = 0
        kappa = 0.0001
        nu = 1
        sigma = 1

        # dimensionality of data
        d = 5

        # sample synthetic features for each label
        # If you want to sample synthetic data from a different
        # Normal-Inverse-Chi-Squared distribution,
        # change kappa, mu, nu, and sigma
        #synth = synthetic.SampleSynthetic(kind='ell', d=d, mu0=mu, kappa0=kappa, nu0=nu, sigma0=sigma)
        #synth.fit()

        #print(synth)

        nix = Priors.NIX2(mu0=mu, kappa0=kappa, nu0=nu, sigma0=sigma)


        # fit the ddCRP model
        # once fitted, crp.map_z_ is the MAP label
        crp = ddCRP.ddCRP(alpha, model=nix, mcmc_passes=30, stats_interval=200, n_clusters=10)



        #print(X)

        adj_list = {i: [j for j in range(len(ids)) if distance_matrix[i,j] != 1] for i in range(len(ids))}


        crp.fit(X, adj_list)




        clusters = crp.map_z_

        #clusters, link, lhood = ddcrp_infer(obs, lhood_fn, distance_, delay, n_iter, alpha)

        print(clusters)

    if version == 0: #SPHERICAL K-MEANS

        #INITIALISATION DES CENTROIDES
        ids_by_frames = {}
        for row in input:
            if row[0] not in ids_by_frames.keys():
                ids_by_frames[row[0]] = []
            ids_by_frames[row[0]].append(row[1])
        n_ids_by_frames = {k: len(v) for k, v in ids_by_frames.items()}
        ref_frame = max(n_ids_by_frames, key=lambda key: n_ids_by_frames[key])
        centers_ids_init=ids_by_frames[ref_frame]
        centers_init = [list(data[:,1]).index(i) for i in centers_ids_init]
        print(data[:,1])
        print(ref_frame, centers_ids_init, centers_init)

        ids = list(np.unique(data[:, 1]))
        if (len(ids) < n_clusters):
            n_clusters = len(ids)



        clusters, centers = cop_kmeans(dataset=data[:,2:], initialization=centers_init+[13],  k=n_clusters, ml=must_link, cl=cannot_link, spherical=True)

        if clusters == None:
            print("Error: impossible clustering")
            exit()



    output = input[:,:10]
    output[:,1] = np.array([clusters[ids.index(int(x))] for x in input[:,1]])

    print("Output saved to:", output_file)
    np.save(output_file, output)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Post clustering")
    parser.add_argument(
        "--input_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--output_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--n_clusters", help="Number of clusters",
        default=None, required=True)
    parser.add_argument(
        "--version", help="Code version (see post_clustering.py source)",
        default=0)
    parser.add_argument(
        "--max_common_frames", help="Maximum number of common frames (constraint)",
        default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input_file, args.output_file, int(args.max_common_frames), int(args.n_clusters), int(args.version))
