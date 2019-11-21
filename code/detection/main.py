from imageai.Detection import ObjectDetection
import csv, os
detector = ObjectDetection()


vid_name = "ajax_chelsea"


out = open("./../data/"+vid_name+"/det/det.txt","w")
with out:
    writer = csv.writer(out)
    #MOT Challenge format
    #<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    N = len(os.listdir("./../data/"+vid_name+"/img1"))
    for f in range(N):
        if f%100==0: print(str(f)+" / "+str(N))
        model_path = "./models/yolo-tiny.h5"
        input_path = "./../data/"+vid_name+"/img1/frame"+str(f)+".jpg"
        output_path = "./output/frame"+str(f)+".jpg"

        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(model_path)
        detector.loadModel()
        detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path, minimum_percentage_probability=30)

        for eachItem in detection:
            if(eachItem["name"]=="person"):
                loc = eachItem["box_points"]
                loc[2] = loc[2] - loc[0]
                loc[3] = loc[3] - loc[1]
                writer.writerow([f,-1]+loc+[eachItem["percentage_probability"]/100,-1,-1,-1])


    out.close()