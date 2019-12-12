import numpy as np
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from copkmeans.cop_kmeans import cop_kmeans


run_common_frames = False

def run(input_file, output_file, max_common_frames, n_clusters):
    input = np.load(input_file)
    #Frame_id, track id, ., ., ., ., ., ., ., ., appearance features
    ids = list(np.unique(input[:,1]))
    print("Total number of ids:",len(ids))
    data = []
    for i in ids:
        data.append(input[input[:,1]==i][:,10:].mean(axis=0))
    data = np.array(data)


    common_frames = np.zeros((len(ids),len(ids)))
    if run_common_frames:
        print("Computing common frames matrix...")
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                n_common_frames = len(set(list(input[input[:, 1] == ids[i]][:, 0])).intersection(list(input[input[:, 1] == ids[j]][:, 0])))
                common_frames[i,j] = n_common_frames
                common_frames[j,i] = n_common_frames
        print("Saved common frames matrix")
        np.save("common_frames.npy", common_frames)
    else:
        print("Loaded common frames matrix")
        common_frames = np.load("common_frames.npy")

    print("Computing cannot link constraints with max_common_frames = ",max_common_frames)
    must_link = []
    cannot_link = [(i,j) for i in range(len(ids)) for j in range(len(ids)) if common_frames[i,j] > max_common_frames ]
    print("Number of constraints", len(cannot_link))
    #cannot_link=[]
    clusters, centers = cop_kmeans(dataset=data, k=n_clusters, ml=must_link, cl=cannot_link)
    print("Clusters", clusters)


    #kmeans = KMeans(n_clusters=n_clusters, n_init=1, init='k-means++', max_iter=3000).fit(data)

    output = input[:,:10]
    output[:,1] = np.array([clusters[ids.index(int(x))] for x in input[:,1]])
    print("Output:", output)
    print("Output saved to:", output_file)
    np.savetxt(output_file, output, delimiter=',')

    #frame_id, track_id, bbox1, bbox2, bbox3, bbox4
    #Regrouper les input par tracklets et associer un vecteur de features a chacun d'eux
    #Mean of the appearance descriptor
    #Skeleton features
    #MSER, SIFT regions
    #Clustering

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
        "--max_common_frames", help="Number of clusters",
        default=240)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input_file, args.output_file, args.max_common_frames, int(args.n_clusters))
