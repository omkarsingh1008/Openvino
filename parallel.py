import cv2
from urtils import load,preprocess,postprocess
import numpy as np
import argparse
def main(filename_path,source):
 
    exec_net,input_layer,output_layer,size = load(filename_path,num_sources=2)
    vid = cv2.VideoCapture(int(source[0]))
    vid1 = cv2.VideoCapture(int(source[1]))
    while(True):
        
        ret, frame = vid.read()
        ret, frame1 = vid1.read()
        input_image = preprocess(frame,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:input_image})
        input_image1 = preprocess(frame1,size)
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0][0]
        infer_res = exec_net.start_async(request_id=1,inputs={input_layer:input_image1})
        frame = postprocess(frame,results)
        status=infer_res.wait()
        results1 = exec_net.requests[1].outputs[output_layer][0][0]
        frame1 = postprocess(frame1,results1)
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