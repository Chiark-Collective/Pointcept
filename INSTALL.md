# install 

The installation can be done via a docker container or locally, if you have configured python, cuda, and all other dependencies already.

## 1a docker container (recommended)

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

## 1b. host environment (advanced)

Assuming you have docker, poetry and python 3.11 installed (for this I use pyenv), do:

```sh
poetry install
poetry shell
pip install flash_attn  # poetry doesn't play well with this dependency
```

and you should be able to run anything relevant inside the poetry environment.

If you're in doubt about local configuration, use the container where everything is taken care of for you!

## 1c. building pointops and pointgroup_ops (advanced - should not be needed for regular use)

These pointcept libraries are already compiled to wheels that are included in this repository under `/wheels`.
Should you need to recompile them (unlikely but possible), you'll need to use the docker container for this purpose that matches the relevant cuda/torch etc dependencies appropriately.
A make alias is included to cover this.

```sh
make copy-wheels
```

this will deposit the newly compiled wheels in `/wheels_container` where you can copy them over the `/wheels` where poetry expects to find them.
You may wish to commit these binaries (they're small enough that git LFS would be excessive).

## 2. PointTransformerV3 model

To clone the point transformer V3 model repo as a git submodule, including pretrained model weights, we'll pull the model weights from the Pointcept huggingface repo.

Note that you have to add your SSH key to huggingface first. 
Huggingface provides full instructions on how to do this:
- https://huggingface.co/docs/hub/en/security-git-ssh



```sh
make ptv3-update
```

This will take some time (7.4G).

Also note that if you're using the docker container, run this *outside* the container, and docker-compose will mount the relevant files for you.

If you have run `make clean` and de-initialised the submodule, you can re-add it with

```sh
make ptv3-add
```
You should be now ready to follow [INSTRUCTIONS.md](INSTRUCTIONS.md) to preprocess and train your heritage data.

# 3. Pointcept datasets

If you want to run any testing or inference with the datasets used to train the pre-trained Pointcept models, a nonexhaustive selection is as follows:

## S3DIS

preprocessed data aligned + with normals (recommended from readme):
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/ERtd0QAyLGNMs6vsM4XnebcBseQ8YTL0UTrMmp11PmQF3g?e=MsER95

OR just aligned (and/or raw):
https://cvg-data.inf.ethz.ch/s3dis/

Unzipped to `./data/s3dis`

## ScanNet v2

preprocessed (from readme):
https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EREuB1If2DNEjz43-rdaVf4B5toMaIViXv8gEbxr9ydeYA?e=ffXeG4

OR raw:
http://www.scan-net.org/

Unzipped to `./data/scannet`.

# refs

[Point Transformer V3](https://arxiv.org/abs/2312.10035)
[PPT](https://arxiv.org/abs/2308.09718)
