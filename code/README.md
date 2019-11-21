Pour lancer le tracker sur la vidéo test:


Aller dans deepsort/

Executer: 


python deep_sort_app.py \
    --sequence_dir=../data/ajax_chelsea \
    --detection_file=./resources/detections/foot/ajax_chelsea.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
