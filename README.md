## Multi-camera person re-identification (self supervised)

person re-identification :- Person re-identification (ReID), identifying a person of interest at other time or place, is a challenging task in computer vision. Its applications range from tracking people across cameras to searching for them in a large gallery, from grouping photos in a photo album to visitor analysis in a retail store. 

## installtion and setup

### openvino installation 
offical [link](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)

after installing openvino we need to Download pretrain model from openvino model zoo or convert model to IR
### download model from openvino model zoo
```bash
sudo ./downloader.py --name person-reidentification-retail-0287
```
### convert model to IR
```bash
python3 /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /media/omkar/new_model/saved_model --transformations_config /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.4.json --tensorflow_object_detection_api_pipeline_config /media/omkar/new_model/pipeline.config --reverse_input_channels
```

## setup 

setup on loacl machine.

Clone repo and install requirements.txt in a Python>=3.6.0 environment, including PyTorch>=1.7.
```bash
git clone https://github.com/omkarsingh1008/Openvino.git
```
## demo
demo on video

```bash
python3 parallel.py --weight person-detection-retail-0013.xml --source video_path
```
if you have multiple source you can add 

```bash
python3 parallel.py --weight person-detection-retail-0013.xml --source 0 2
```

## perosn detection demo

https://user-images.githubusercontent.com/48081267/162435914-4e4f81e4-78a6-472e-85e2-590a14b9dabc.mp4


## re identification demo 

https://user-images.githubusercontent.com/48081267/167352316-88a81018-b561-4264-8c30-3d7d1c5ed884.mp4


