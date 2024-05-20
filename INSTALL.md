# install 

## environment

Assuming you have docker, poetry and python 3.11 installed (for this I use pyenv), do:

```sh
make install
```

This will build the necessary CUDA wheels for `pointops`, `pointgroup_ops` and then create a python
environment with these and all other dependencies.

## PointTransformerV3 model

To clone the point transformer V3 model repo as a git submodule, including pretrained model weights:

```sh
make ptv3
```

Note that you have to add your SSH key to huggingface first. This will take some time (7.4G).

# datasets

## S3DIS

preprocessed data aligned + with normals (recommended from readme):
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/ERtd0QAyLGNMs6vsM4XnebcBseQ8YTL0UTrMmp11PmQF3g?e=MsER95

OR just aligned (and/or raw):
https://cvg-data.inf.ethz.ch/s3dis/

I unzipped em in ./data/s3dis

## ScanNet v2

preprocessed (from readme):
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EREuB1If2DNEjz43-rdaVf4B5toMaIViXv8gEbxr9ydeYA?e=ffXeG4

OR raw:
http://www.scan-net.org/

Unzipped in ./data/scannet


# Test the PTV3 model on ScanNet

# test
```sh
poetry run python tools/test.py --config-file ${PTV3_CONFIG_PATH} --options save_path=${PTV3_SAVE_PATH} weight=${PTV3_WEIGHTS_PATH}
```

# refs

[Point Transformer V3](https://arxiv.org/abs/2312.10035)
[PPT](https://arxiv.org/abs/2308.09718)
