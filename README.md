# InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling

 <p align="center">
    <a href="https://gohar-malik.github.io/intermask/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2410.10010"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2410.10010-b31b1b.svg"></a>
    <a href="https://paperswithcode.com/sota/motion-synthesis-on-interhuman?p=intermask-3d-human-interaction-generation-via"><img alt="pwc_interhuman" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/intermask-3d-human-interaction-generation-via/motion-synthesis-on-interhuman"></a>
   <a href="https://paperswithcode.com/sota/motion-synthesis-on-inter-x?p=intermask-3d-human-interaction-generation-via"><img alt="pwc_interx" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/intermask-3d-human-interaction-generation-via/motion-synthesis-on-inter-x"></a>
  </p>

![teaser_image](assets/teaser.jpg)
If you find our code or paper helpful, please consider starring our repository and citing:
```
@inproceedings{javed2025intermask,
  title={InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling},
  author={Muhammad Gohar Javed and Chuan Guo and Li Cheng and Xingyu Li},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=ZAyuwJYN8N}
}
```

## :round_pushpin: Preparation

<details>
  
### 1. Setup Environment
```
conda env create -f environment.yml
conda activate intermask
```
The code was tested on Python 3.7.7 and PyTorch 1.13.1

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.py
```

#### Download Evaluation Models
For evaluation only. Obtained from the [InterGen github repo](https://github.com/tr3e/InterGen/tree/master).
```
bash prepare/download_evaluator.sh
```

The download scripts use the [gdown](https://github.com/wkentaro/gdown) package. If you face problems try running the following command and try again. Solution is from this [github issue](https://github.com/wkentaro/gdown/issues/110#issuecomment-913561232).
```
rm -f ~/.cache/gdown/cookies.json
```

### 3. Get Data

#### InterHuman
Follow the instructions in the [InterGen github repo](https://github.com/tr3e/InterGen/tree/master?tab=readme-ov-file#2-get-data) to download the InterHuman dataset and place it in the `./data/InterHuman/` foler and unzip the `motions_processed.zip` archive such that the directory structure looks like:
```
./data
├── InterHuman
    ├── annots
    ├── LICENSE.md
    ├── motions
    ├── motions_processed
    └── split
```
</details>

## :rocket: Demo
<details>

```
python infer.py --gpu_id 0 --dataset_name interhuman --name trans_default
```

The inference script obtains text prompts from the file `./prompts.txt`. The format is each text prompt per line. By default the script generateds motion of 3 seconds in length. In our work, motion is in 30 fps.

The output files are stored under folder `./checkpoints/<dataset_name>/<name>/animation_infer/`, which is this case would be `./checkpoints/interhuman/trans_default/animation_infer/`. The output files are organized as follows:
* `keypoint_npy`: generated motions with shape of (nframe, 22, 9) for each interacting individual, under subfolder.
* `keypoint_mp4`: stick figure animation in mp4 format, with two viewpoints.

We also apply naive foot ik to the generated motions, see files with prefix `ik_`. It sometimes works well, but sometimes will fail.
</details>