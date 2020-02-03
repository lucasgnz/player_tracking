Pour faire un tracking sur une nouvelle vidéo, il suffit d'appeler main.py pour faire tourner les différents modules / étapes qui sont activées ou désactivées manuellement dans main.py (par des if True: ou if False:).

Placer les images dans data/nom_de_la_video/img1/frameX.jpg, ou X est le numéro de la frame.

Appeler les scripts dans main.py dans l'ordre suivant (les différentes sorties des modules sont utilisées en entrées dans les suivants):


1. Générer les détections à partir des images (génère un fichier data/nom_de_la_video/det/det.txt)


2. Extraires les features à partir de ces détections (génère un fichier deep_sort/resources/detections/foot/nom_de_la_video_det.npy)



3. A) DEEP SORT + CLUSTERING A POSTERIORI

    a. Deep Sort (génère un fichier data/nom_de_la_video/det_ds/det.txt)
    b. Clustering post Deep Sort (génère un fichier data/nom_de_la_video/det_ds_pc/det.txt) 


OU


3. B) DEEP SORT LIMIT (version modifiée de Deep Sort limitant le nombre total d'identifiants générés)




4. Visualiser le résultat en vidéo
