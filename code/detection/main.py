from imageai.Detection import ObjectDetection
import csv, os
detector = ObjectDetection()



def detection(path, model_path, offset, len_):
    out = open(path+"/det/det.txt","w")
    with out:
        writer = csv.writer(out)
        #MOT Challenge format
        #<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        N = len(os.listdir(path+"/img1"))
        if offset+len_> N:
            print("offset + len > N")
            return
        for f in range(len_):
            if f%400==0: print(str(f)+" / "+str(len_))
            input_path = path+"/img1/frame"+str(offset+f)+".jpg"
            output_path = path+"/det/frame"+str(f)+".jpg"

            detector.setModelTypeAsTinyYOLOv3()
            detector.setModelPath(model_path)
            detector.loadModel()
            detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path, minimum_percentage_probability=15)

            for eachItem in detection:
                if(eachItem["name"]=="person"):
                    loc = eachItem["box_points"]
                    loc[2] = loc[2] - loc[0]
                    loc[3] = loc[3] - loc[1]
                    writer.writerow([f,-1]+loc+[eachItem["percentage_probability"]/100,-1,-1,-1])


        out.close()
        print("Detections stored in "+path+"/det/det.txt")
        return

if __name__ == "__main__":
    detection("test_short", 6827)