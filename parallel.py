import cv2
from urtils import load,preprocess,postprocess,draw_tracks,load_reid
import numpy as np
import argparse
from motrackers import CentroidTracker,SORT,IOUTracker
from reid import reidentification_P
#from motrackers.utils import draw_tracks
def main(filename_path,source):
    exec_net,input_layer,output_layer,size = load_reid("/media/omkar/omkar3/openvino/openvino_parallel/Openvino/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml")
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    tracker1 = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    vid = cv2.VideoCapture("2.mp4")
    vid1 = cv2.VideoCapture("demo1.mp4")
    ids = {}
    ids1 = {}
    track_id={}
    while(True):
        
        ret, frame = vid.read()
        ret, frame1 = vid1.read()
        h,w = size[2:]
        frame = cv2.resize(frame, (w,h))
        frame1 = cv2.resize(frame1, (w,h))
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
        #print(bboxes1,scores1,labels1)
        tracks = tracker.update(bboxes, scores,labels)
        tracks1 = tracker1.update(bboxes1, scores1,labels1)
        #print(tracks)

        frame,ids = draw_tracks(frame, tracks,ids)
        frame1,ids1 = draw_tracks(frame1, tracks1,ids1)
        print(ids.keys())
        print(ids1.keys())
        reidentification_P(ids,ids1,track_id,frame,frame1)

        frame = cv2.resize(frame, (500, 500))
        frame1 = cv2.resize(frame1, (500, 500))
        frame = np.hstack([frame,frame1])

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