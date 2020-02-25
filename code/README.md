Pour faire un tracking sur une nouvelle vidéo, il suffit d'appeler main.py pour faire tourner les différents modules / étapes qui sont activées ou désactivées manuellement au début du fichier. La configuration des différents hyperparamètres se trouve également au début de main.py. Les paramètres de l'extraction des features LOMO sont dans lomo_config.json.

Placer les images dans data/nom_de_la_video/img1/frameX.jpg, ou X est le numéro de la frame.

Remplir la configuration dans main.py.

Activer les scripts dans main.py dans l'ordre suivant (les différentes sorties des modules sont utilisées en entrées dans les suivants):


1. Générer les détections à partir des images (génère un fichier data/nom_de_la_video/det/det.txt)


2. Extraires les features à partir de ces détections (génère un fichier deep_sort/resources/detections/foot/nom_de_la_video_det{config.fe}.npy)


3.a) Générer les tracklets avec Deep Sort (génère un fichier data/nom_de_la_video/det{config.fe}_ds{config.ds}/det.npy)
    
3.b) Clustering post Deep Sort (génère un fichier data/nom_de_la_video/det{config.fe}_ds{config.ds}_pc{config.pc}/det.npy) pour réduire le nombre de tracklets au nombre voulu (nombre de joueurs présents sur la vidéo)

OU

3. Générer les tracklets avec Limited Deep Sort (génère un fichier data/nom_de_la_video/det{config.fe}_dsl{config.dsl}/det.npy)

4. Visualiser les différents résultats en vidéo et les sauvegarder dans des fichiers .avi.





Pour effectuer un tuning des hyperparamètres, activer model_selection et modifier la fin du fichier main.py (la partie utilisant hyperopt). Il faut également activer les modules que l'on veut inclure dans les modèles à comparer.
