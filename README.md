# ✋SRHand: Super-Resolving Hand Images and 3D Shapes via View/Pose-aware Neural Image Representations and Explicit 3D Meshes
### NIPS 2025


[Minje Kim](https://yunminjin2.github.io), [Tae-Kyun Kim](https://sites.google.com/view/tkkim/home)

[![report](https://img.shields.io/badge/Project-Page-blue)](https://yunminjin2.github.io/projects/srhand/)
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://yunminjin2.github.io/projects/srhand/)
<p align='center'>
    <img src='assets/TeaserVideo.gif'/>
</p>

> Reconstructing detailed hand avatars plays a crucial role in various applications. While prior works have focused on capturing high-fidelity hand geometry, they heavily rely on high-resolution multi-view image inputs and struggle to generalize on low-resolution images. Multi-view image super-resolution methods have been proposed to enforce 3D view consistency. These methods, however, are limited to static objects/scenes with fixed resolutions and are not applicable to articulated deformable hands. In this paper, we propose SRHand (Super-Resolution Hand), the method for reconstructing detailed 3D geometry as well as textured images of hands from low-resolution images. SRHand leverages the advantages of implicit image representation with explicit hand meshes. Specifically, we introduce a geometric-aware implicit image function (GIIF) that learns detailed hand prior by upsampling the coarse input images. By jointly optimizing the implicit image function and explicit 3D hand shapes, our method preserves multi-view and pose consistency among upsampled hand images, and achieves fine-detailed 3D reconstruction (wrinkles, nails). In experiments using the InterHand2.6M and Goliath datasets, our method significantly outperforms state-of-the-art image upsampling methods adapted to hand datasets, and 3D hand reconstruction methods, quantitatively and qualitatively. The code will be publicly available.

&nbsp;


## Environmental Setting

```
conda create -n srhand python=3.9
```
1. Install torch.
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

2. Install pytorch3d.
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

3. Install additional requirements.
```
pip install -r requirements.txt
```

4. Download Interhand2.6M (5 fps) dataset and set the dataset paths in config file in conf folder.

## Training GIIF

```
python main_giif.py --conf conf/giif_16_256_128.conf

python main_disc.py --conf conf/disc_16_256_128.conf 
```
## Validating GIIF
```
python main_disc.py --conf conf/disc_16_256_128.conf --eval --model_path PATH_TO_GIIF --model_path_disc PATH_TO_DISCRIMINATOR
```

(e.g)
```
python main_disc.py --conf conf/disc_16_256_128.conf --eval  --model_path ./interhand_out/Capture0_ROM03_RT_No_Occlusion/GIIF_16_256/liif.pth --model_path_disc ./interhand_out/Capture0_ROM03_RT_No_Occlusion/Disc_GIIF_16_256/disc.pth
```

## Training SRHand

After you have finished traning GIIF, you can now train SRHand.

```
python main.py --conf conf/ih_GIIF_16_256.conf
```

## Validating SRHand
To validate SRHand, use below code.

'''
python main.py --conf conf/ih_GIIF_test.conf --eval --r --model_path FOLDER_PATH_EXPERIMENT
'''

(e.g)
```
python main.py --conf conf/ih_GIIF_test.conf --eval --r --model_path ./interhand_out/Capture0_ROM03_RT_No_Occlusion/SRHand_16 
```

To save visualization result, give —save_vis option.




## Citation

If you find this work useful, please consider citing our paper.

```
@InProceedings{kim2025srhand,
    author = {Kim, Minje and Kim, Tae-Kyun},
    title = {SRHand: Super-Resolving Hand Images and 3D Shapes via View/Pose-aware Nueral Image Representations and Explicit 3D Meshes},
    booktitle = {Advances in Neural Information Processing Systems (NIPS)},
    year = {2025}
}
```

&nbsp;

## Acknowledgements
 - Our code is based on [XHand](https://github.com/agnJason/XHand).
 - We also thank the authors of [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/), [Goliath](https://github.com/facebookresearch/goliath) for the useful dataset.
 - The renderer are based on the renderer from [nvdiffrast](https://github.com/NVlabs/nvdiffrast). 
