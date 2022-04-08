import cv2
from openvino.inference_engine import IECore
ie = IECore()
def load(filename,num_sources = 1):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)

def preprocess(frame,size):
    n,c,h,w = size
    input_image = cv2.resize(frame, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image

def postprocess(frame,results):
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        h1, w1 = frame.shape[:2]
        
        boxes.append(tuple(map(int, (xmin * w1, ymin * h1, xmax * w1, ymax * h1))))
        labels.append(int(label))
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=0.6, nms_threshold=0.7)
    

    boxes=[boxes[idx] for idx in indices]

    for  box in boxes:
       
        cv2.rectangle(img=frame, pt1=box[:2], pt2=box[2:], color=(0,255,0), thickness=3)
    return frame
