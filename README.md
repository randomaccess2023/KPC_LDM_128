# Implementation of a Conditional Latent Diffusion-Based Generative Model to Synthetically Create Unlabeled Histopathological Images

The code for the paper titled _**Implementation of a Conditional Latent Diffusion-Based Generative Model to Synthetically Create Unlabeled Histopathological Images**_ has been shared in this _GitHub_ repository.

I will explain different _**folders**_ and _**files**_ available in this repository very briefly to make things easy for the users.

- ### Let's focus on the `clusters_14` folder first:
1. `part_1_training.ipynb`: This file has the code for the _first_ training of the **cLDM**.
2. `part_2_finetuning1.ipynb`: This file features the code for the _finetuning-1_ (additional training-1) of the **cLDM**.
3. `part_3_finetuning2.ipynb`: This file houses the code for the _finetuning-2_ (additional training-2) of the **cLDM**.
4. `part_4_sampling_clustering.ipynb`: This file offers the code for clustering the training images using the **cLDM** obtained after finetuning-2. The sampling part has also been performed with a few selected test samples using the _second_ finetuned **cLDM**.
5. `part_5_latent_space_extraction.ipynb`: This file provides the implementation for saving the latent space in a `.csv` file to use later for performing _**internal cluster validation**_ using a _second_ finetuned **Classifier**.
6. `part_6_conditional_generation.ipynb`: This file encompasses the code for generating 10,000 images from the conditioning of 10,000 training images using the **cLDM** obtained after finetuning-2.
7. `part_7_sampling_after_first_training.ipynb`: This file includes the source code to carry out sampling using the conditioning of a few selected test samples after the _first_ training of the **cLDM**.
8. `part_8_sampling_after_finetuning1.ipynb`: This file hosts the code to do sampling with a few selected test samples using the **cLDM** obtained _finetuning-1_.
9. `part_9_conditional_generation_using_cluster_IDs.ipynb`: This file incorporates the code for generating similar samples seen in the clustering output of the training images using only cluster IDs.
10. `part_10_SSIM`: This file stores the code for calculating the **SSIM** metric using both the original (training) and generated images.
11. `part_11_MSSSIM`: This file holds the codebase for quantifying the **MS-SSIM** metric using both the original (training) and generated images.
12. `part_12_LPIPS`: This file contains the code for computing the **LPIPS** metric using both the original (training) and generated images.

>> In addition to these 12 files, there is a folder called **models_14** within **clusters_14** which has two models: `classifier_finetuning2_ckpt_20250128_70_14.pth` and `classifier_training_ckpt_20250127_600_14.pth`. Three more models are necessary for executing the 12 `.ipynb` files shown above: `unet_finetuning1_ckpt_20250128_130_14.pth`, `unet_finetuning2_ckpt_20250128_70_14.pth`, and `unet_training_ckpt_20250127_600_14.pth`. Each of these 3 file has a size of around 93 MB, and, therefore, they couldn't be uploaded to this repository.

>>> These 3 UNet models can be found [here](https://doi.org/10.6084/m9.figshare.29588807). There is a `.zip` file and by unpacking, a folder called `trained_models` can be observed. Inside this folder, the 3 UNet models are present. These 3 UNet models will have to be moved in the **`models_14`** folder.

One more file is available in this folder: `HHH14_C14_D14.csv`. This is the same `.csv` file that we mentioned in `part_5_latent_space_extraction.ipynb`.

>> An additional information: Viewers can see this address in multiple files -> **'/project/dsc-is/nono/Documents/kpc/dat0'**, which points to the location of our dataset used for this experiment in our JupyterHub server -> **'slice128_Block2_11K.npy'**. This is a _NumPy_ file and it has a size of 4.33 GB. We couldn't upload it here. Users can set up this address according to their working environment.

#### We will provide update in the future if our dataset can be accessed somehow.

>> Finally, for calculating image evaluation metrics like `SSIM`, `MS-SSIM`, and `LPIPS`, training image directory (**'org'**) and generated image directory (in this case, **'gen_14'**) are required. These directories contain 10,000 images each, and together they have a size of approximately 132 MB.

>>> The directories can be downloaded from [here](https://doi.org/10.6084/m9.figshare.29588849). There is a `.zip` file named **`Images.zip`**. By unzipping, two folders can be found in the `Images` folder: `org` and `gen_14`. Put the `gen_14` folder inside the _**`clusters_14`**_ directory and the `org` folder in the _**`KPC_LDM_128`**_ directory.

_We only shared the code for **clusters_14**, but the same thing can be repeated for clusters 10, 11, 12, 13, 15, and 16 in six separate folders. Only a change in the variable named `n_clusters` needs to be made according to the desired number of clusters_.

- ### The folder `Figures` has 4 folders inside it: `Figure 1`, `Figure 2`, `Figure 3`, and `Figure 4`. These 4 folders have the figures that can be seen in the journal.

- ### The folder `MODULE` is needed to calculate internal cluster validation indices: _Calinski-Harabasz index_, _C index_, _Dunn index_, _Hartigan index_, and _Mclain-Rao index_ utilizing the latent features.

Check the script `internal_cluster_validation_metrics.ipynb` to see the deatils.

>> Since we shared only the **clusters_14** folder, people can only see `HHH14_C14_D14.csv` file in this repository. Other required files like `HHH10_C10_D10.csv`, `HHH11_C11_D11.csv`, `HHH12_C12_D12.csv`, `HHH13_C13_D13.csv`, `HHH15_C15_D15.csv`, and `HHH16_C16_D16.csv` can be obtained by repeating the works shown in the **clusters_14** folder for **clusters_10**, **clusters_11**, **clusters_12**, **clusters_13**, **clusters_15**, and **clusters_16** folder.

- ### The folder `kpc_ldm` has the autoencoder model: _`vqvae_autoencoder_ckpt.pth`_. This pre-trained autoencoder is used for applying the diffusion and denoising processes in the latent space of the images rather than the pixel space.

- ### The folder `weights` has the pre-trained weights of the VGG-16 model in the _`vgg.pth`_ file. These weights were used both in the _LPIPS model_ and _LPIPS metric_.

- ### Let's explain the last folder `pfiles`:

It has many Python (`.py`) files.
1. `blocks.py`: This file encompasses the building blocks for the autoencoder and the UNet.
2. `discriminator.py`: This file has the code for the discriminator model.
3. `linear_noise_scheduler.py`: This file holds the codebase for adding noise using a linear variance scheduler on latent features.
4. `lpips.py`: This file incorporates the code for implementing the LPIPS model.
5. `lpips_metric.py`: This file contains the code for computing the LPIPS metric.
6. `unet_cond_base.py`: This file features the code for creating the UNet architecture. It uses functions from `blocks.py` file.
7. `vqvae.py`: This file offers the code for constructing the autoencoder architecture. It uses functions from `blocks.py` file.

> Finally, `train_autoencoder.ipynb` script includes the source code for training the autoencoder. `autoencoder_output.ipynb` file displays the code for obtaining reconstructed samples after training the autoencoder. `save_training_images.ipynb` shows the code for saving training images in a directory for evaluating image quality later.

## This repository uses `MIT License`. Read the terms and conditions from _LICENSE_ text file.

# Read the paper from here: [KPC_LDM](https://www.mdpi.com/2306-5354/12/7/764)
