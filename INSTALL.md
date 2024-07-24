# install 

The installation can be done via a docker container or locally, if you have configured python, cuda, and all other dependencies already.

## docker container

To install the container run

```sh
docker-compose up pointcept-env -d
```

or just

```sh
make docker-env
```

to get into a bash container inside the container:

```sh
docker exec -it pointcept-env /bin/bash
```

or just

```sh
make enter-container
```

## host environment

Assuming you have docker, poetry and python 3.11 installed (for this I use pyenv), do:

```sh
poetry install
poetry shell
pip install flash_attn  # poetry doesn't play well with this dependency
```

and you should be able to run anything relevant inside the poetry environment. If you're in doubt about this, use the container where everything is taken care of for you.

## building pointops and pointgroup_ops (advanced)

These pointcept libraries are already compiled to wheels that are included in this repository under `/wheels`.
Should you need to recompile them (unlikely but possible), you'll need to use the docker container for this purpose that matches the relevant cuda/torch etc dependencies appropriately.
A make alias is included to cover this.

```sh
make copy-wheels
```

this will deposit the newly compiled wheels in `/wheels container` where you can copy them over the `/wheels` where poetry expects to find them.
You may wish to commit these binaries (they're small enough that git LFS would be excessive).

## PointTransformerV3 model

To clone the point transformer V3 model repo as a git submodule, including pretrained model weights:

```sh
make ptv3
```

Note that you have to add your SSH key to huggingface first. This will take some time (7.4G).

Also note that if you're using the docker container, run this *outside* the container, and docker-compose will mount the relevant files for you.

# datasets

## S3DIS

preprocessed data aligned + with normals (recommended from readme):
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/ERtd0QAyLGNMs6vsM4XnebcBseQ8YTL0UTrMmp11PmQF3g?e=MsER95

OR just aligned (and/or raw):
https://cvg-data.inf.ethz.ch/s3dis/

I unzipped em to `./data/s3dis`

## ScanNet v2

preprocessed (from readme):
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EREuB1If2DNEjz43-rdaVf4B5toMaIViXv8gEbxr9ydeYA?e=ffXeG4

OR raw:
http://www.scan-net.org/

Unzipped to `./data/scannet`.


<!-- # Test the PTV3 model on ScanNet

# test
```sh
make test-ptv3 TODO: needs fixed
``` -->

# refs

[Point Transformer V3](https://arxiv.org/abs/2312.10035)
[PPT](https://arxiv.org/abs/2308.09718)
