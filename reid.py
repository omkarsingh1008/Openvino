import cv2
from openvino.inference_engine import IECore
from urtils import load_reid,preprocess
ie = IECore()
import torch
import matplotlib.pyplot as plt

exec_net,input_layer,output_layer,size = load_reid("/media/omkar/omkar3/openvino/openvino_parallel/Openvino/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml")


def reidentification_P(ids,ids1,track_id,frame,frame1):
    if len(track_id)==0:
        for i,bbox in ids.items():
            img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            plt.imshow(img)
            plt.show()
            print("ffff")
            img = preprocess(img,size)
            infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
            status=infer_res.wait()
            results = exec_net.requests[0].outputs[output_layer][0]
            for j,bbox1 in ids1.items():
                print("gggg")
                img1 = frame1[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
                #plt.imshow(img1)
                #plt.show()
                img1 = preprocess(img1,size)
                infer_res = exec_net.start_async(request_id=1,inputs={input_layer:img1})
                status=infer_res.wait()
                results1 = exec_net.requests[1].outputs[output_layer][0]
                print(results1.shape)
                x = torch.tensor(results).unsqueeze(0)
                y = torch.tensor(results1).unsqueeze(0)
                dis=torch.cosine_similarity(x, y)[0].numpy()
                if dis >= 0.5:
                    track_id[i]= bbox


                

def reidentification(ids,track_id,frame):
    for i,bbox in ids.items():
        if len(track_id) == 0:
            track_id[i] = bbox
        else:
            for j,bbox_t in track_id.items():
                img = bbox[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                img1 = bbox_t[bbox_t[1]:bbox_t[3], bbox_t[0]:bbox_t[2]]
                img = preprocess(frame,size)
                infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
                status=infer_res.wait()
                results = exec_net.requests[0].outputs[output_layer][0]

