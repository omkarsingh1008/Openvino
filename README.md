# Openvino
## Openvino parallel inference
# openvino installation 
offical [link](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)

after installing model we need to Download pretrain model from openvino model zoo or convert model to IR
## download model from openvino model zoo
```bash
sudo ./downloader.py --name person-reidentification-retail-0287
```
## convert model to IR
```bash
python3 /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /media/omkar/DATA/Darsa/vehicle_detection/new_model/new_model/saved_model --transformations_config /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.4.json --tensorflow_object_detection_api_pipeline_config /media/omkar/DATA/Darsa/vehicle_detection/new_model/new_model/pipeline.config --reverse_input_channels
```

## setup 

setup on loacl machine.

Clone repo and install requirements.txt in a Python>=3.6.0 environment, including PyTorch>=1.7.
```bash
git clone https://github.com/omkarsingh1008/product_detection_from_shelf.git
```
## demo
demo on video

```bash
python3 parallel.py --weight person-detection-retail-0013.xml --source video_path
```
if you have multiple source you can add 

```bash
python3 parallel.py --weight person-detection-retail-0013.xml --source video_path1 video_path2
```

https://user-images.githubusercontent.com/48081267/162435914-4e4f81e4-78a6-472e-85e2-590a14b9dabc.mp4



