from glob import glob
import cv2
from urtils import load,preprocess,postprocess,draw_tracks,load_reid,draw_rec
import numpy as np
import argparse
from motrackers import CentroidTracker,SORT,IOUTracker
from reid import reidentification_P,reidentification,emb
import numpy as np
import cv2
from collections import deque
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances as ed
import zmq
import os
import torch

track_id={}
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=(256,), activation="relu"),
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(0.001))

embs = deque(maxlen=10)
label = deque(maxlen=10)

def manage_cluster(vect,bbox):
    final = []
    for v, bb in zip(vect,bbox):
        v=v.reshape(-1,256)
        #print(v.shape)
        prediction = model.predict(v)
        idc = np.argmax(prediction)
        score = prediction[0][idc]
        final.append([score, idc, bb, v])
    final.sort(key=lambda x: x[0], reverse=True)
    assigned = []
    as_return = []
    randoms = [0, 1, 2, 3, 4, 5, 6, 7,8,9]
    for score, idc, bb, vector in final:
        if idc in assigned:
            new_id = np.random.choice(randoms)
            randoms.remove(new_id)
            assigned.append(new_id)
            idc = new_id
        else:
            assigned.append(idc)
            randoms.remove(idc)
        embs.append(vector[0])
        op = tf.keras.utils.to_categorical(idc, num_classes=10)
        label.append(op)
        xt = np.array(embs)
        yt = np.array(label)
        if score >= 0.20:
            model.fit(xt, yt, verbose=0, epochs=1)
        as_return.append([score, idc, bb])
    return as_return
#from motrackers.utils import draw_tracks
def main(filename_path,source):
    global track_id
    exec_net,input_layer,output_layer,size = load_reid("/media/omkar/omkar3/openvino/openvino_parallel/Openvino/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml")
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    tracker1 = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    vid = cv2.VideoCapture("4.mp4")
    vid1 = cv2.VideoCapture("4.mp4")
    grads = []
    uu = []
    while(True):
        
        ret, frame = vid.read()
        ret, frame1 = vid1.read()
        ids = {}
        ids1 = {}
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
        #print(ids.keys())
        #print(ids1.keys())
        #cv2.imshow('frameq', frame)
        
        vect,bbox=emb(ids,frame)
        #print(vect[0].shape)
        c = manage_cluster(vect,bbox)
        #print(c)
        for score, idx, bb in c:
            if idx in uu:
                index_ = uu.index(idx)
                previous_bb = grads[index_]
                distance = ed([previous_bb], [bb])[0][0]
                if distance <= 5:
                    bb = previous_bb
                grads[index_] = bb
            else:
                uu.append(idx)
                grads.append(bb)
            final_bb = []
            for x in bb:
                final_bb.append(int(x))
            bb = final_bb
            bb = np.array(bb)
            x_left, x_top, x_right, x_bottom = bb
        # print(bb)
            cv2.putText(frame, "ID: " + str(idx), (x_left, x_top), 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
            cv2.rectangle(frame, (x_left, x_top), (x_right, x_bottom), (0, 255, 0), 1)


        #print(track_id)
        #draw_img = draw_rec(frame,track_id)
        
        draw_img = cv2.resize(frame, (500, 500))
        #frame1 = cv2.resize(frame1, (500, 500))
        #frame = np.hstack([frame,frame1])

        cv2.imshow('frame', draw_img)
        
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