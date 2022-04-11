import cv2
from urtils import load,preprocess,postprocess,draw_tracks
import numpy as np
import argparse
from motrackers import CentroidTracker,SORT,IOUTracker
#from motrackers.utils import draw_tracks
def main(filename_path,source):
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    vid = cv2.VideoCapture(int(source[0]))
    vid1 = cv2.VideoCapture("2.mp4")
    ids = {}
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
        tracks = tracker.update(bboxes1, scores1,labels1)
        #print(tracks)
    
        frame1 = draw_tracks(frame1, tracks)
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