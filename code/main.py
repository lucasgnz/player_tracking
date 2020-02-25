import os
import numpy as np
import pandas as pd
from detection.main import detection
from sklearn.decomposition import PCA
import pprint

config = {}
config['fe'] = {}
config['ds'] = {}
config['dsl'] = {}
config['pc'] = {}
config['pca'] = {}


def print_line():
    print("------------------------------------------------------------------------------------------------------------------------------------------------------")

###########################################    DEBUT CONFIGURATION     #################################################
########################################################################################################################
#General configuration
config['path'] = "/Users/lucasgonzalez/work/telecom/PRIM/"
config['vid_name'] = "100"
config['offset'] = 500
config['n_frames'] = 4500



#Configuration - Feature extraction
config['fe']['alpha_op'] = 0.0
config['fe']['lomo_config'] = "lomo_config.json"

#Configuration - PCA
config['pca']['active'] = False
config['pca']['ndim'] = 300

#Configuration - Deep Sort
config['ds']['max_age'] = 50
config['ds']['n_init'] = 50
config['ds']['max_iou_distance'] = 0.7
config['ds']['min_confidence'] = 0.10
config['ds']['max_cosine_distance'] = 0.25
config['ds']['nn_budget'] = 100
config['ds']['alpha_ds'] = 0.0



#Configuration - Limited Deep Sort
config['dsl']['n_init'] = 30
config['dsl']['max_iou_distance'] = 0.7
config['dsl']['min_confidence'] = 0.10
config['dsl']['max_cosine_distance'] = 0.25
config['dsl']['nn_budget'] = 100
config['dsl']['metric_param'] = 0.03
config['dsl']['alpha_ds'] = 0.0
config['dsl']['max_tracks'] = 11


#Configuration - Post clustering
config['pc']['version'] = 3
config['pc']['n_clusters'] = 10
config['pc']['max_common_frames'] = 0

###########################################    FIN CONFIGURATION     ###################################################
########################################################################################################################


###################################    ACTIVATION / DESACTIVATION MODULES     ##########################################
########################################################################################################################

detection_ = 0#Première étape, detection image par image

#ATTENTION l'extracteur agit sur tous les dossiers dans data, (pour lancer l'extraction sur plusieurs vidéos à la fois)
#AVANT DE LANCER CE MODULE, rajouter un . devant les noms de dossiers dans data/ qui ont déjà été traitées pour les masquer
feature_extraction = 0 #Lancer l'extraction des features avec la configuration dans config['fe']


#Activer ou non le pca, le nombre de features final est à régler plus bas dans le bloc   if pca_:...
pca_ = 0

deep_sort = 0 #Lancer deepsort sur les données correspondant à la configuration en cours
visualize_ds = 0 #Visualiser les tracklets produits par deep sort

post_clustering = 0 #Lancer le post clustering sur les données correspondant à la configuration en cours
visualize_pc = 0 #Visualiser les tracklets produits par le clustering post deepsort

deep_sort_limit = 0 #Lancer la version modifiée de deepsort sur les données correspondant à la configuration en cours
visualize_dsl = 0 #Visualiser les tracklets produits par la versoin modifiée de deep sort


score = 1 #Afficher les scores pour la configuration en cours

model_selection = 1


visualize_gt = 0 #Visualiser les annotations (groundtruth)

#################################    FIN ACTIVATION / DESACTIVATION MODULES     ########################################
########################################################################################################################

new_configs = ['version', 'alpha_ds', 'max_tracks']
former_default = [0, 0.0, 10]


def update_config_str(config):
    if 'str' in config.keys():
        del config['str']
    config['str'] = "_"+str(config).replace(" ","").replace("'","")
    for c in config.values():
        if isinstance(c, dict):
            if 'str' in c.keys():
                del c['str']

            c_=c.copy()
            for (k,d) in zip(new_configs, former_default):
                if k in c.keys() and c[k] == d:
                    del c_[k]
            c['str'] = "_"+str(c_).replace(" ","").replace("'","")
    return config

update_config_str(config)


def run(config, optimized_stat='N_a'):
    print("Config: ", config['str'])
    print_line()
    #####   SCORE AGAINST GROUND TRUTH   #####
    def score_det(verbose=0):
        os.system("python3 " + config['path'] + "code/score.py \
            --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
            --verbose=" + str(verbose)+ " \
            --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+("_pca" if config['pca']['active'] else '')+"/det.npy' \
            --gt_file='" + config['path'] + "data/" + config['vid_name'] + "/gt/gt.npy' \
            --offset=" + str(config['offset']))
    def score_ds(verbose=0):
        os.system("python3 " + config['path'] + "scode/core.py \
            --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
            --verbose=" + str(verbose) + " \
            --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+("_pca" if config['pca']['active'] else '') + "_ds" +
                  config['ds']['str'] + "/det.npy' \
            --gt_file='" + config['path'] + "data/" + config['vid_name'] + "/gt/gt.npy' \
            --offset=" + str(config['offset']))
    def score_dsl(verbose=0):
        os.system("python3 " + config['path'] + "code/score.py \
            --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
            --verbose=" + str(verbose) + " \
            --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+("_pca" if config['pca']['active'] else '') + "_dsl" +
                  config['dsl']['str'] + "/det.npy' \
            --gt_file='" + config['path'] + "data/" + config['vid_name'] + "/gt/gt.npy' \
            --offset=" + str(config['offset']))

    def score_pc(verbose=0):
        os.system("python3 " + config['path'] + "code/score.py \
            --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
            --verbose=" + str(verbose) + " \
            --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+("_pca" if config['pca']['active'] else '') + "_ds" +
                  config['ds']['str'] + "_pc" + config['pc']['str'] + "/det.npy' \
            --gt_file='" + config['path'] + "data/" + config['vid_name'] + "/gt/gt.npy' \
            --offset=" + str(config['offset']))


    #####   DETECTION   #####
    if detection_:
        print("Detection...")
        detection(config['path']+"data/"+config['vid_name'], config['path']+"code/detection/models/yolo-tiny.h5", config['offset'], config['n_frames'])
        score_det()

    if feature_extraction:
        print("Features extraction from detection boxes "+("'and open pose data/'" if config['fe']['alpha_op'] > 0 else '')+"...")
        os.system("python3 "+config['path']+"code/deep_sort/tools/generate_detections.py \
        --model="+config['path']+"code/deep_sort/resources/networks/mars-small128.pb \
        --mot_dir="+config['path']+"data/ \
        --offset="+str(config['offset'])+" \
        --det_stage='det"+config['fe']['str']+"' \
        --openpose="+("'openpose/'" if config['fe']['alpha_op'] > 0 else '')+" \
        --alpha_op="+str(config['fe']['alpha_op'])+"\
        --lomo_config=" + config['path'] + str(config['fe']['lomo_config'])+" \
        --output_dir='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+"'")

        print("Detections + Features stored in "+ config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+"/det.npy")

    # PCA
    if pca_ and config['pca']['active']:
        print("Run PCA...")
        detections = np.load(config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+"/det.npy")
        print("Data shape: ", detections[:, 10:].shape)
        pca = PCA(n_components=config['pca']['ndim'])
        detections_pca = pca.fit_transform(detections[:, 10:])

        detections_ = np.empty((detections.shape[0], 310))
        detections_[:, :10] = detections[:, :10]
        detections_[:, 10:] = detections_pca

        detections = detections_
        print("After PCA: ", detections[:, 10:].shape)

        np.save(config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+"_pca/det.npy", detections)
        print("PCA features stored in "+config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+"_pca/det.npy")


    #####   DEEPSORT   #####
    if deep_sort:
        print("Running Deep Sort algorithm...")
        if not os.path.exists(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"/"):
            os.mkdir(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"/")
        os.system("python3 "+config['path']+"code/deep_sort/deep_sort_app.py \
        --sequence_dir="+config['path']+"data/"+config['vid_name']+"/ \
        --detection_file='"+ config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+("_pca" if config['pca']['active'] else '')+"/det.npy' \
        --offset="+str(config['offset'])+" \
        --n_frames=" + str(config['n_frames']) + " \
        --max_iou_distance=" + str(config['ds']['max_iou_distance']) + " \
        --max_age=" + str(int(config['ds']['max_age'])) + " \
        --alpha_ds=" + str(config['dsl']['alpha_ds']) + "\
        --n_init=" + str(int(config['ds']['n_init'])) + " \
        --min_confidence=" + str(config['ds']['min_confidence']) + " \
        --max_cosine_distance=" + str(config['ds']['max_cosine_distance']) + "\
        --nn_budget=" + str(int(config['ds']['nn_budget'])) + " \
        --output_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"/det.npy' \
        --display=False")
        print("Deep Sort tracklets stored in "+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"/det.npy")
        score_ds()

    #####   POST CLUSTERING   #####
    if post_clustering:
        print("Post-clustering of Deep Sort tracklets...")
        if not os.path.exists(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/"):
            os.mkdir(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/")
        os.system("python3  "+config['path']+"code/post_clustering" + (str(int(config['pc']['version'])) if 'version' in config['pc'].keys() else "0")+".py \
        --input_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"/det.npy' \
        --output_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/det.npy' \
        --max_common_frames=" + str(config['pc']['max_common_frames']) + " \
        --n_clusters=" + str(config['pc']['n_clusters']) + " \
        --version=" + (str(config['pc']['version']) if 'version' in config['pc'].keys() else "0"))
        print("Clustered tracklets stored in "+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+("_pca" if config['pca']['active'] else '')+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/det.npy")
        score_pc()

    #####  MODIFIED / LIMITED DEEPSORT   #####
    if deep_sort_limit:
        print("Running Modified/Limited Deep Sort algorithm...")
        if not os.path.exists(config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
        "_pca" if config['pca']['active'] else '') + "_dsl" + config['dsl']['str'] + "/"):
            os.mkdir(config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
                "_pca" if config['pca']['active'] else '') + "_dsl" + config['dsl']['str'] + "/")
        os.system("python3 " + config['path'] + "code/deep_sort_limit/deep_sort_app.py \
                --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
                --detection_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str']+("_pca" if config['pca']['active'] else '') + "/det.npy' \
                --offset=" + str(config['offset']) + " \
                --n_frames=" + str(config['n_frames']) + " \
                --max_iou_distance=" + str(config['dsl']['max_iou_distance']) + " \
                --max_age=10000000000 \
                --max_tracks="+str(int(config['dsl']['max_tracks']))+" \
                --n_init=" + str(int(config['dsl']['n_init'])) + " \
                --min_confidence=" + str(config['dsl']['min_confidence']) + " \
                --max_cosine_distance=" + str(config['dsl']['max_cosine_distance']) + "\
                --metric_param=" + str(config['dsl']['metric_param']) + "\
                --alpha_ds=" + str(config['dsl']['alpha_ds']) + "\
                --nn_budget=" + str(int(config['dsl']['nn_budget'])) + " \
                --output_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
                      "_pca" if config['pca']['active'] else '') + "_dsl" + config['dsl']['str'] + "/det.npy' \
                --display=False")
        print("Limited Deep Sort tracklets stored in " + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe'][
            'str'] + ("_pca" if config['pca']['active'] else '') + "_dsl" + config['dsl']['str'] + "/det.npy")
        score_dsl()

    #####   VISUALIZE   #####
    if visualize_pc:
        os.system("python3 " + config['path'] + "code/deep_sort/show_results.py \
           --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
           --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
            "_pca" if config['pca']['active'] else '') + "_ds" + config['ds']['str'] + "_pc" + config['pc']['str'] + "/det.npy' \
           --offset=" + str(config['offset']) + " \
           --output_file='" + config['path'] + "results/" + config['vid_name'] + config['fe']['str'] + (
                      "_pca" if config['pca']['active'] else '') + "_ds" + config['ds']['str'] + "_pc" + config['pc']['str'] + ".avi' \
           --update_ms=41")

    #####   VISUALIZE DEEPSORT TRACKLETS #####
    if visualize_ds:
        os.system("python3 " + config['path'] + "code/deep_sort/show_results.py \
           --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
           --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
            "_pca" if config['pca']['active'] else '') + "_ds" +
                  config['ds']['str'] + "/det.npy' \
           --offset=" + str(config['offset']) + " \
           --output_file='" + config['path'] + "results/" + config['vid_name'] + config['fe']['str'] + (
            "_pca" if config['pca']['active'] else '') + "_ds" +
                  config['ds']['str'] + "_pc" + config['pc']['str'] + ".avi' \
           --update_ms=41")
    #####    VISUALIZE LIMITED DEEPSORT TRACKLETS
    if visualize_dsl:
        os.system("python3 " + config['path'] + "code/deep_sort/show_results.py \
           --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
           --result_file='" + config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
            "_pca" if config['pca']['active'] else '') + "_dsl" +
                  config['dsl']['str'] + "/det.npy' \
           --offset=" + str(config['offset']) + " \
           --output_file='" + config['path'] + "results/" + config['vid_name'] + config['fe']['str'] + (
            "_pca" if config['pca']['active'] else '') + "_dsl" +
                  config['dsl']['str'] + "_pc" + config['pc']['str'] + ".avi' \
           --update_ms=41")

    #####   VISUALIZE GROUND TRUTH   #####
    if visualize_gt:
        os.system("python3 " + config['path'] + "code/deep_sort/show_results.py \
           --sequence_dir=" + config['path'] + "data/" + config['vid_name'] + "/ \
           --result_file='" + config['path'] + "data/" + config['vid_name'] + "/gt/gt.npy' \
           --offset=" + str(config['offset']) + " \
           --output_file='" + config['path'] + "results/" + config['vid_name'] + "_gt.avi' \
           --update_ms=41")

    det_score_path = config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
        "_pca" if config['pca']['active'] else '') + "/det_score.csv"
    ds_score_path = config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
        "_pca" if config['pca']['active'] else '') + "_ds" + config['ds']['str'] + "/det_score.csv"
    dsl_score_path = config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
        "_pca" if config['pca']['active'] else '') + "_dsl" + config['dsl']['str'] + "/det_score.csv"
    pc_score_path = config['path'] + "data/" + config['vid_name'] + "/det" + config['fe']['str'] + (
        "_pca" if config['pca']['active'] else '') + "_ds" + config['ds']['str'] + "_pc" + config['pc'][
                        'str'] + "/det_score.csv"

    if score:
        score_ds()
        score_det()
        score_pc()
        score_dsl()
        print_line()
        scores = pd.DataFrame()
        def add_score(model, s, scores):
            s['Model'] = model
            if scores.shape[0] == 0:
                scores = s
            else:
                scores = pd.concat((scores,s))
            return scores

        if os.path.exists(det_score_path):
            scores = add_score("DETECTION", pd.read_csv(det_score_path), scores)
        if os.path.exists(ds_score_path):
            scores = add_score("DEEPSORT", pd.read_csv(ds_score_path), scores)
        if os.path.exists(pc_score_path):
            scores = add_score("DEEPSORT + POST CLUSTERING", pd.read_csv(pc_score_path), scores)
        if os.path.exists(dsl_score_path):
            scores = add_score("LIMITED DEEPSORT", pd.read_csv(dsl_score_path), scores)

        pprint.pprint(scores[['Model', 'Purity', 'test', 'Number of IDs', 'N_a', 'recall', 'mota', 'motp', 'idf1']])

        pprint.pprint(scores[['Model', 'Purity', 'Number of IDs', 'N_a', 'recall','precision', 'idp', 'idr', 'idf1',
                                    'mota', 'motp']])

        print_line()

        pprint.pprint(scores[['Model', 'num_predictions', 'num_matches', 'num_objects', 'num_switches','num_ascend','num_transfer','num_migrate']])

    if deep_sort:
        return -pd.read_csv(ds_score_path)[optimized_stat].values[0]
    if deep_sort_limit:
        return -pd.read_csv(dsl_score_path)[optimized_stat].values[0]
    if post_clustering:
        return -pd.read_csv(pc_score_path)[optimized_stat].values[0]
    if detection_:
        return -pd.read_csv(det_score_path)[optimized_stat].values[0]

    return 0


#MODEL SELECTION
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

if model_selection:
    # define an objective function
    def objective(args, optimized_stat):
        print_line()
        args = update_config_str(args)
        try:
            s = run(args, optimized_stat=optimized_stat)
            return s
        except:
            print("Erreur avec la configuration suivante : {}".format(args))
            return np.inf

    # define a search space
    space = config.copy()


    space['dsl']['max_cosine_distance'] = hp.uniform('max_cosine_distance', 0.10, 0.40)
    space['dsl']['nn_budget'] = hp.quniform('nn_budget', 80, 200, 1)
    space['dsl']['n_init'] = hp.quniform('n_init', 10, 70, 1)
    space['dsl']['metric_param'] = hp.uniform('metric_param', 0.005, 0.15)
    space['dsl']['alpha_ds'] = hp.uniform('alpha_ds', 0, 1)

    best = {'alpha_ds': 0.8460598636064135, 'max_cosine_distance': 0.15411527411190198,
            'metric_param': 0.046651398971124136, 'n_init': 24.0, 'max_tracks':10, 'nn_budget': 210.0}#A recalculer


    for k, v in best.items():
        space['dsl'][k] = v


    optimized_stat = 'idf1'

    # minimize the objective over the space
    best = fmin(lambda x: objective(x, optimized_stat), space, algo=tpe.suggest, max_evals=1)

    print(best)
else:
    run(config)



"""#Configuration - Deep Sort
config['ds']['max_age'] = 50
config['ds']['n_init'] = 50
config['ds']['max_iou_distance'] = 0.7
config['ds']['min_confidence'] = 0.10
config['ds']['max_cosine_distance'] = 0.25
config['ds']['nn_budget'] = 100
config['ds']['alpha_ds'] = 0.0


#Configuration - Limited Deep Sort
config['dsl']['n_init'] = 30
config['dsl']['max_iou_distance'] = 0.7
config['dsl']['min_confidence'] = 0.10
config['dsl']['max_cosine_distance'] = 0.25
config['dsl']['nn_budget'] = 100
config['dsl']['metric_param'] = 0.03
config['dsl']['alpha_ds'] = 0.0
config['dsl']['max_tracks'] = 11


#Configuration - Post clustering
config['pc']['version'] = 3
config['pc']['n_clusters'] = 11
config['pc']['max_common_frames'] = 0"""