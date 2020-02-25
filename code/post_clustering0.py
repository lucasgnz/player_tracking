import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from copkmeans.cop_kmeans import cop_kmeans
import time


import operator




run_common_frames = True

def run(input_file, output_file, max_common_frames, n_clusters, version):
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



    min_len_tracklet = 0

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
        d[2:] = d[2:] / np.linalg.norm(d[2:]) * n_frames
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
    #print(common_frames, common_frames.shape)

    print("Computing 'cannot link' constraints with max_common_frames = ",max_common_frames)
    must_link = []
    cannot_link = [(i,j) for i in range(len(ids)) for j in range(i+1, len(ids)) if common_frames[i,j] > max_common_frames]
    #print(cannot_link)
    print("Number of constraints", len(cannot_link))


    #INITIALISATION DES CENTROIDES
    init_mode = 2

    ids_by_frames = {}
    for row in input:
        if row[0] not in ids_by_frames.keys():
            ids_by_frames[row[0]] = []
        ids_by_frames[row[0]].append(row[1])



    if init_mode == 1 or True:#On cherche la frame ayant le plus de joueurs détectés simultanément,
        # et on initialise les clusters avec les tracklets détectés sur cette frame
        n_ids_by_frames = {k: len(v) for k, v in ids_by_frames.items()}
        ref_frame = max(n_ids_by_frames, key=lambda key: n_ids_by_frames[key])
        print(ref_frame, ids_by_frames[ref_frame], [list(data[:, 1]).index(i) for i in ids_by_frames[ref_frame]])


    if init_mode == 2: # On cherche la frame sur laquelle le somme des longueurs des tracklets détectés sur cette frame est maximale
        # et on initialise les clusters avec les tracklets détectés sur cette frame
        len_by_frames = {}
        for row in input:
            if row[0] not in len_by_frames.keys():
                len_by_frames[row[0]] = []
            len_by_frames[row[0]].append(np.linalg.norm(data[data[:, 1] == row[1]].reshape(-1)[2:]))
        sum_len_by_frames = {k: sum(v) for k, v in len_by_frames.items()}
        ref_frame = max(sum_len_by_frames, key=lambda key: sum_len_by_frames[key])

    centers_ids_init = ids_by_frames[ref_frame]


    centers_init = [list(data[:, 1]).index(i) for i in centers_ids_init]

    print(ref_frame, centers_ids_init, centers_init)

    ids = list(np.unique(data[:, 1]))
    if (len(ids) < n_clusters):
        n_clusters = len(ids)


    clusters, centers = cop_kmeans(dataset=data[:,2:], initialization=centers_init,  k=n_clusters, ml=must_link, cl=cannot_link, spherical=True)

    if clusters == None:
        print("Error: impossible clustering")
        exit()

    print([clusters[ids.index(int(x))] for x in centers_ids_init])

    #clusters = [np.random.randint(0,n_clusters) for i in range(len(ids))] #RANDOM CLUSTERING

    clusters_={2.0: [1.0], 584.0: [1.0], 644.0: [1.0], 910.0: [1.0], 1060.0: [1.0], 1435.0: [1.0], 1593.0: [1.0], 1732.0: [1.0], 2021.0: [1.0], 2149.0: [1.0], 2273.0: [1.0], 2455.0: [1.0], 2550.0: [1.0], 2671.0: [1.0], 2680.0: [1.0, 6.0], 21.0: [2.0], 67.0: [2.0], 103.0: [2.0], 270.0: [2.0], 346.0: [2.0], 399.0: [2.0], 666.0: [2.0], 1129.0: [2.0], 1382.0: [2.0], 1658.0: [2.0], 1714.0: [2.0], 2029.0: [2.0], 2087.0: [2.0], 2348.0: [2.0], 2785.0: [2.0], 224.0: [3.0], 283.0: [3.0], 316.0: [3.0], 386.0: [3.0], 936.0: [3.0], 1299.0: [3.0], 1374.0: [3.0], 1462.0: [3.0], 1567.0: [3.0], 1636.0: [3.0], 1700.0: [3.0], 1860.0: [3.0], 2432.0: [3.0], 2594.0: [3.0], 2643.0: [3.0], 2711.0: [3.0], 4.0: [4.0], 575.0: [4.0], 676.0: [4.0], 756.0: [4.0], 888.0: [4.0], 950.0: [4.0], 1156.0: [4.0, 6.0], 1918.0: [4.0, 8.0], 2086.0: [4.0], 2163.0: [4.0], 2358.0: [4.0], 2553.0: [4.0], 3.0: [5.0], 180.0: [5.0], 235.0: [5.0], 304.0: [5.0], 401.0: [5.0], 893.0: [5.0], 1428.0: [5.0], 1558.0: [5.0], 2032.0: [5.0], 2743.0: [5.0], 1.0: [6.0], 42.0: [6.0], 140.0: [6.0], 181.0: [6.0], 510.0: [6.0], 560.0: [6.0], 686.0: [6.0], 819.0: [6.0], 961.0: [6.0], 1058.0: [6.0], 1403.0: [6.0], 1990.0: [6.0], 2059.0: [6.0], 2253.0: [6.0], 2416.0: [6.0], 2516.0: [6.0], 2748.0: [6.0], 5.0: [7.0], 108.0: [7.0], 262.0: [7.0], 344.0: [7.0], 438.0: [7.0], 504.0: [7.0], 578.0: [7.0], 832.0: [7.0], 941.0: [7.0], 1038.0: [7.0], 1407.0: [7.0], 1547.0: [7.0], 1783.0: [7.0], 1866.0: [7.0], 1910.0: [7.0], 2222.0: [7.0], 2458.0: [7.0], 2570.0: [7.0], 2645.0: [7.0], 6.0: [8.0], 54.0: [8.0], 89.0: [8.0], 261.0: [8.0], 334.0: [8.0], 409.0: [8.0], 440.0: [8.0], 637.0: [8.0], 932.0: [8.0], 1102.0: [8.0], 1173.0: [8.0], 1406.0: [8.0], 1508.0: [8.0], 1644.0: [8.0], 2007.0: [8.0], 2166.0: [8.0], 2480.0: [8.0], 2525.0: [8.0], 46.0: [9.0], 661.0: [9.0], 894.0: [9.0], 1029.0: [9.0], 1325.0: [9.0], 1831.0: [9.0], 1973.0: [9.0], 2268.0: [9.0], 2139.0: [10.0], 1355.0: [11.0]}

    #output = np.array([x for x in input[:,:10] if x[1] in clusters_.keys()])
    #output[:,1] = np.array([clusters_[x][0] for x in output[:,1]])

    output = input[:, :10]
    output[:, 1] = np.array([clusters[ids.index(int(x))] for x in output[:, 1]])

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
