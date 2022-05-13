import cv2
from urtils import load,preprocess,postprocess
import numpy as np
import argparse
from motrackers import CentroidTracker
from urtils import load,draw_tracks,tracklets,tracking
from reid import ids_feature_,distance_,distance_list,ids_feature_list
from multiprocessing.pool import ThreadPool
import time
pool = ThreadPool(processes=2)

import threading

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

def nothing(x):
    pass
def main(filename_path,source):
 
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    tracker1 = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    vid = cv2.VideoCapture((source[0]))
    vid1 = cv2.VideoCapture((source[1]))
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    frame_size = (frame_width,frame_height)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output2.mp4', fourcc, fps, (1000, 500))
    #cv2.namedWindow('Multi camera tracking')
    #cv2.createTrackbar('track_id','Multi camera tracking',0,100,nothing)
    #cv2.createTrackbar('ON/OFF','Multi camera tracking',0,1,nothing)
    tracks_id={}
    tracklets_id={}
    prev_frame_time = 0
    new_frame_time = 0

    while(True):
        ids = {}
        ids1={}
        tracks_draw={}
        tracks_draw1={}
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
        #async_result = pool.apply_async(tracking, (ids,frame,tracks_id,tracklets_id,tracks_draw))
        #tracks_draw,tracks_id,tracklets_id = async_result.get()
        #async_result1 = pool.apply_async(tracking, (ids1,frame1,tracks_id,tracklets_id,tracks_draw1))
        #tracks_draw1,tracks_id,tracklets_id = async_result1.get()
        tracks_draw,tracks_id,tracklets_id=tracking(ids,frame,tracks_id,tracklets_id,tracks_draw)
        #tracks_draw1,tracks_id,tracklets_id=tracking(ids1,frame1,tracks_id,tracklets_id,tracks_draw1)
        #print(tracks_id.keys())
        #print(tracklets_id)
        #thread1 = ThreadWithResult(target=tracking, args=(ids,frame,tracks_id,tracklets_id,tracks_draw))
        #thread2 = ThreadWithResult(target=tracking, args=(ids1,frame1,tracks_id,tracklets_id,tracks_draw1))
        #thread1.start()
        #thread2.start()
        #thread1.join()
        #thread2.join()
        #print(thread1.tracks_draw,thread1.tracks_id,thread1.tracklets_id)
        #print(thread2.result)


            
        for id,bbox in tracks_draw.items():
            cv2.putText(frame, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 255, 0), 1)
        #for id,bbox in tracks_draw1.items():
        #    cv2.putText(frame1, str(id), bbox[:2], 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
        #    cv2.rectangle(frame1, bbox[:2], bbox[2:], (0, 255, 0), 1)
        #frame = cv2.resize(frame, (500,500))
        #frame1 = cv2.resize(frame1, (500,500))
        #frame = np.hstack([frame,frame1])

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        
        output.write(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Multi camera tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()
    vid1.release()
    output.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--weight',default="")
    args.add_argument('-s', '--source', required=True, nargs='+',
                        help='Input sources (indexes of cameras or paths to video files)')
    parsed_args=args.parse_args()
    main(filename_path=parsed_args.weight,source=parsed_args.source)