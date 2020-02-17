import os
import numpy as np


config = {}
config['fe'] = {}
config['ds'] = {}
config['pc'] = {}

######## DEBUT CONFIGURATION #########

#General configuration
config['path'] = "/Users/lucasgonzalez/work/telecom/PRIM/code/"
config['vid_name'] = "100"
config['offset'] = 500
config['n_frames'] = 4500

#Configuration - Feature extraction
config['fe']['alpha_op'] = 1.0
config['fe']['lomo_config'] = "lomo_config.json"

#Configuration - Deep Sort
config['ds']['max_age'] = 50
config['ds']['n_init'] = 50
config['ds']['max_iou_distance'] = 0.7
config['ds']['min_confidence'] = 0.10
config['ds']['max_cosine_distance'] = 0.25
config['ds']['nn_budget'] = 100

#Configuration - Post clustering
config['pc']['n_clusters'] = 11
config['pc']['max_common_frames'] = 0


#Activer / Désactiver les différents modules
detection_ = False

#AVANT DE LANCER CE MODULE, RAJOUTER un . DEVANT LES NOMS DE VIDEOS DANS data/ QUI ONT DEJA ETE TRAITEES,
feature_extraction = False

deep_sort = True
post_clustering = True



visualize = True


deep_sort_limit = False

######## FIN CONFIGURATION #########



config['str'] = "_"+str(config).replace(" ","").replace("'","")
for c in config.values():
    if isinstance(c, dict):
        c['str'] = "_"+str(c).replace(" ","").replace("'","")



print(config['fe']['str'])
print(config['str'])


#####   DETECTION   #####
from detection.main import detection
if detection_:
    print("Detection...")
    detection(config['path']+"data/"+config['vid_name'], config['path']+"detection/models/yolo-tiny.h5", config['offset'], config['n_frames'])





if feature_extraction:
    print("Features extraction from detection boxes and open pose data...")
    os.system("python3 "+config['path']+"deep_sort/tools/generate_detections.py \
    --model="+config['path']+"deep_sort/resources/networks/mars-small128.pb \
    --mot_dir="+config['path']+"data/ \
    --offset="+str(config['offset'])+" \
    --det_stage='det"+config['fe']['str']+"' \
    --openpose=openpose/ \
    --alpha_op="+str(config['fe']['alpha_op'])+"\
    --lomo_config=" + config['path'] + str(config['fe']['lomo_config'])+" \
    --output_dir="+config['path']+"deep_sort/resources/detections/foot")

    print("Detections + Features stored in "+config['path']+"deep_sort/resources/detections/foot/"+config['vid_name']+"_det"+config['fe']['str']+".npy")

#####   DEEPSORT   #####
if deep_sort:
    print("Running Deep Sort algorithm...")
    if not os.path.exists(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"/"):
        os.mkdir(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"/")
    os.system("python3 "+config['path']+"deep_sort/deep_sort_app.py \
    --sequence_dir="+config['path']+"data/"+config['vid_name']+"/ \
    --detection_file='"+config['path']+"deep_sort/resources/detections/foot/"+config['vid_name']+"_det"+config['fe']['str']+".npy' \
    --offset="+str(config['offset'])+" \
    --n_frames=" + str(config['n_frames']) + " \
    --max_iou_distance=" + str(config['ds']['max_iou_distance']) + " \
    --max_age=" + str(config['ds']['max_age']) + " \
    --n_init=" + str(config['ds']['n_init']) + " \
    --min_confidence=" + str(config['ds']['min_confidence']) + " \
    --max_cosine_distance=" + str(config['ds']['max_cosine_distance']) + "\
    --nn_budget=" + str(config['ds']['nn_budget']) + " \
    --output_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"/det.txt' \
    --display=False")
    print("Deep Sort tracklets stored in "+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"/det.txt")



#####   POST CLUSTERING   #####
if post_clustering:
    print("Post-clustering of Deep Sort tracklets...")
    if not os.path.exists(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/"):
        os.mkdir(config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/")
    os.system("python3  "+config['path']+"deep_sort/post_clustering.py \
    --input_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"/det.txt' \
    --output_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/det.txt' \
    --max_common_frames=" + str(config['pc']['max_common_frames']) + " \
    --n_clusters=" + str(config['pc']['n_clusters']) + "")

    print("Clustered tracklets stored in "+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/det.txt")


#####   VISUALIZE   #####
if visualize:
    os.system("python3 "+config['path']+"deep_sort/show_results.py \
    --sequence_dir="+config['path']+"data/"+config['vid_name']+"/ \
    --result_file='"+config['path']+"data/"+config['vid_name']+"/det"+config['fe']['str']+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+"/det.txt' \
    --offset="+str(config['offset'])+" \
    --output_file='"+config['path']+"results/"+config['vid_name']+config['fe']['str']+"_ds"+config['ds']['str']+"_pc"+config['pc']['str']+".avi' \
    --update_ms=41")




#####  MODIFIED / LIMITED DEEPSORT   #####
if deep_sort_limit:
    print("Running Modified/Limited Deep Sort algorithm...")
    if not os.path.exists(config['path']+"data/"+config['vid_name']+"/det_dsl/"):
        os.mkdir(config['path']+"data/"+config['vid_name']+"/det_dsl/")
    os.system("python3 "+config['path']+"deep_sort_limit/deep_sort_app.py \
    --sequence_dir="+config['path']+"data/"+config['vid_name']+"/ \
    --detection_file="+config['path']+"deep_sort/resources/detections/foot/"+config['vid_name']+"_det.npy \
    --offset="+str(config['offset'])+" \
    --n_frames=" + str(config['n_frames']) + " \
    --max_iou_distance=0.7 \
    --max_age=5000 \
    --n_init=50 \
    --max_tracks=10 \
    --metric_param=0 \
    --min_confidence=0.25 \
    --max_cosine_distance=0.25 \
    --nn_budget=1000 \
    --output_file="+config['path']+"data/"+config['vid_name']+"/det_dsl/det.txt \
    --display=True")
    print("Deep Sort 10 tracklets stored in "+config['path']+"data/"+config['vid_name']+"/det_dsl/det.txt")



#####   VISUALIZE   #####
if False:
    os.system("python3 "+config['path']+"deep_sort/show_results.py \
    --sequence_dir="+config['path']+"data/"+config['vid_name']+"/ \
    --result_file="+config['path']+"data/"+config['vid_name']+"/det_dsl/det.txt \
    --offset="+str(config['offset'])+" \
    --output_file="+config['path']+"results/"+config['vid_name']+"_dsl.avi \
    --update_ms=41 \
    --detection_file="+config['path']+"deep_sort/resources/detections/foot/"+config['vid_name']+"_det.npy")





### AVANT DE LANCER CE MODULE, RAJOUTER un . DEVANT LES NOMS DE DOSSIERS DANS data/ QUI ONT DEJA ETE TRAITES
"""
if False:
    print("Creating detection boxes from openpose data + features extraction from detection boxes...")
    os.system("python3 "+config['path']+"deep_sort/tools/generate_detections_openpose.py \
    --model="+config['path']+"deep_sort/resources/networks/mars-small128.pb \
    --mot_dir="+config['path']+"data/ \
    --offset="+str(offset)+" \
    --openpose='openpose/' \
    --output_dir="+config['path']+"deep_sort/resources/detections/foot")

    print("Detections + Features stored in "+config['path']+"deep_sort/resources/detections/foot/"+config['vid_name']+"_detbyop.npy")
"""
