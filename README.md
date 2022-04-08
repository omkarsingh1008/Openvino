# Openvino

# openvino installation 
offical [link](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
after installing model we need to Download pretrain model from openvino model zoo or convert model to IR
## for downloading model from openvino model zoo
```bash
sudo ./downloader.py --name person-reidentification-retail-0287
```
## conver model to IR
```bash
python3 /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir /media/omkar/DATA/Darsa/vehicle_detection/new_model/new_model/saved_model --transformations_config /opt/intel/openvino_2021.4.582/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.4.json --tensorflow_object_detection_api_pipeline_config /media/omkar/DATA/Darsa/vehicle_detection/new_model/new_model/pipeline.config --reverse_input_channels
```
Openvino parallel inference
