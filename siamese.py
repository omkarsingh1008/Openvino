import cv2
from urtils import load,preprocess,postprocess
import numpy as np
import argparse
from motrackers import CentroidTracker
from urtils import draw_tracks
from reid import ids_feature_,distance_

def main(filename_path,source):
 
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    tracker1 = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    vid = cv2.VideoCapture((source[0]))
    vid1 = cv2.VideoCapture((source[1]))
    tracks_id={}
    while(True):
        ids = {}
        ids1={}
        tracks_draw={}
        ret, frame = vid.read()
        ret, frame1 = vid1.read()
        input_image = preprocess(frame,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:input_image})
        input_image1 = preprocess(frame1,size)
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0][0]
        infer_res = exec_net.start_async(request_id=1,inputs={input_layer:input_image1})
        bboxes, scores,labels,frame = postprocess(frame,results)
        status=infer_res.wait()
        results1 = exec_net.requests[1].outputs[output_layer][0][0]
        bboxes1,scores1,labels1,frame1 = postprocess(frame1,results1)
        tracks = tracker.update(bboxes, scores,labels)
        tracks1 = tracker1.update(bboxes1, scores1,labels1)

        frame1,ids1 = draw_tracks(frame1, tracks1,ids1)
        frame,ids = draw_tracks(frame, tracks,ids)
        frame = np.hstack([frame,frame1])
        #print(ids.keys())
        #print(ids.keys())

        ids_feat = ids_feature_(ids,frame)

        if len(tracks_id)==0:
            tracks_id = ids_feat
            tracks_draw = ids
        else:
            for i,feature in ids_feat.items():
                dis={}
                for id,feat1 in tracks_id.items():
                    d = distance_(feature,feat1)
                    dis[id]=d
                max_key = max(dis, key=dis.get)
                print("first_id:-",i)
                print("track_id:-",id)
                print("reid:-",max_key)
                print("reid_dis:-",dis[max_key])
                print("reid_dis:-",dis)
                
                if dis[max_key] > .5:
                    tracks_id[max_key] = feature
                    tracks_draw[max_key] = ids[i]
                else:
                    tracks_id[len(tracks_id)+1] = feature
                    tracks_draw[len(tracks_id)+1] = ids[i]
        print(tracks_draw)
        print(tracks_id.keys())

        for id,bbox in tracks_draw.items():
            cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)


        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()
    vid1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--weight',default="")
    args.add_argument('-s', '--source', required=True, nargs='+',
                        help='Input sources (indexes of cameras or paths to video files)')
    parsed_args=args.parse_args()
    main(filename_path=parsed_args.weight,source=parsed_args.source)