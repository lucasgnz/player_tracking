import os
import numpy as np


path = "/Users/lucasgonzalez/work/telecom/PRIM/code/"
vid_name = "100"
offset = 500
n_frames = 300

#####   DETECTION   #####
from detection.main import detection




if False:
    print("Detection...")
    detection(path+"data/"+vid_name, path+"detection/models/yolo-tiny.h5", offset, n_frames)


#####   FEATURES EXTRACTION   #####
### AVANT DE LANCER CE MODULE, RAJOUTER un . DEVANT LES NOMS DE DOSSIERS DANS data/ QUI ONT DEJA ETE TRAITES
if False:
    print("Features extraction from detection boxes...")
    os.system("python3 "+path+"deep_sort/tools/generate_detections.py \
    --model="+path+"deep_sort/resources/networks/mars-small128.pb \
    --mot_dir="+path+"data/ \
    --offset="+str(offset)+" \
    --det_stage=det \
    --openpose='' \
    --output_dir="+path+"deep_sort/resources/detections/foot")

    print("Detections + Features stored in "+path+"deep_sort/resources/detections/foot/"+vid_name+"_det.npy")




for alpha_op in [1.0]:#np.arange(5)/5:
    print(alpha_op)
    ### AVANT DE LANCER CE MODULE, RAJOUTER un . DEVANT LES NOMS DE DOSSIERS DANS data/ QUI ONT DEJA ETE TRAITES
    if True:
        print("Features extraction from detection boxes (with open pose data)...")
        os.system("python3 "+path+"deep_sort/tools/generate_detections.py \
        --model="+path+"deep_sort/resources/networks/mars-small128.pb \
        --mot_dir="+path+"data/ \
        --offset="+str(offset)+" \
        --det_stage=det_op"+str(alpha_op)+" \
        --openpose=openpose/ \
        --alpha_op="+str(alpha_op)+"\
        --lomo_config=" + path + "lomo_config.json \
        --output_dir="+path+"deep_sort/resources/detections/foot")

        print("Detections + Features stored in "+path+"deep_sort/resources/detections/foot/"+vid_name+"_det_op_"+str(alpha_op)+".npy")

    #####   DEEPSORT   #####
    if False:
        print("Running Deep Sort algorithm...")
        if not os.path.exists(path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds/"):
            os.mkdir(path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds/")
        os.system("python3 "+path+"deep_sort/deep_sort_app.py \
        --sequence_dir="+path+"data/"+vid_name+"/ \
        --detection_file="+path+"deep_sort/resources/detections/foot/"+vid_name+"_det_op"+str(alpha_op)+".npy \
        --offset="+str(offset)+" \
        --n_frames=" + str(n_frames) + " \
        --max_iou_distance=0.7 \
        --max_age=50 \
        --n_init=50 \
        --min_confidence=0.25 \
        --max_cosine_distance=0.25 \
        --nn_budget=100 \
        --output_file="+path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds/det.txt \
        --display=False")
        print("Deep Sort tracklets stored in "+path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds/det.txt")



    #####   POST CLUSTERING   #####
    if False:
        print("Post-clustering of Deep Sort tracklets...")
        if not os.path.exists(path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds_pc/"):
            os.mkdir(path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds_pc/")
        os.system("python3  "+path+"deep_sort/post_clustering.py \
        --input_file="+path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds/det.txt \
        --output_file="+path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds_pc/det.txt \
        --max_common_frames=0 \
        --n_clusters=10")

        print("10 Tracklets stored in "+path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds_pc/det.txt")


    #####   VISUALIZE   #####
    if False:
        os.system("python3 "+path+"deep_sort/show_results.py \
        --sequence_dir="+path+"data/"+vid_name+"/ \
        --result_file="+path+"data/"+vid_name+"/det_op"+str(alpha_op)+"_ds_pc/det.txt \
        --offset="+str(offset)+" \
        --output_file="+path+"results/"+vid_name+"_op"+str(alpha_op)+"_ds_pc.avi \
        --update_ms=41")









#####  MODIFIED / LIMITED DEEPSORT   #####
if False:
    print("Running Modified/Limited Deep Sort algorithm...")
    if not os.path.exists(path+"data/"+vid_name+"/det_dsl/"):
        os.mkdir(path+"data/"+vid_name+"/det_dsl/")
    os.system("python3 "+path+"deep_sort_limit/deep_sort_app.py \
    --sequence_dir="+path+"data/"+vid_name+"/ \
    --detection_file="+path+"deep_sort/resources/detections/foot/"+vid_name+"_det.npy \
    --offset="+str(offset)+" \
    --n_frames=" + str(n_frames) + " \
    --max_iou_distance=0.7 \
    --max_age=5000 \
    --n_init=50 \
    --max_tracks=10 \
    --metric_param=0 \
    --min_confidence=0.25 \
    --max_cosine_distance=0.25 \
    --nn_budget=1000 \
    --output_file="+path+"data/"+vid_name+"/det_dsl/det.txt \
    --display=True")
    print("Deep Sort 10 tracklets stored in "+path+"data/"+vid_name+"/det_dsl/det.txt")



#####   VISUALIZE   #####
if False:
    os.system("python3 "+path+"deep_sort/show_results.py \
    --sequence_dir="+path+"data/"+vid_name+"/ \
    --result_file="+path+"data/"+vid_name+"/det_dsl/det.txt \
    --offset="+str(offset)+" \
    --output_file="+path+"results/"+vid_name+"_dsl.avi \
    --update_ms=41 \
    --detection_file="+path+"deep_sort/resources/detections/foot/"+vid_name+"_det.npy")





### AVANT DE LANCER CE MODULE, RAJOUTER un . DEVANT LES NOMS DE DOSSIERS DANS data/ QUI ONT DEJA ETE TRAITES
"""
if False:
    print("Creating detection boxes from openpose data + features extraction from detection boxes...")
    os.system("python3 "+path+"deep_sort/tools/generate_detections_openpose.py \
    --model="+path+"deep_sort/resources/networks/mars-small128.pb \
    --mot_dir="+path+"data/ \
    --offset="+str(offset)+" \
    --openpose='openpose/' \
    --output_dir="+path+"deep_sort/resources/detections/foot")

    print("Detections + Features stored in "+path+"deep_sort/resources/detections/foot/"+vid_name+"_detbyop.npy")
"""
