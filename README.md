# LF-Transmission

This is the repository for paper "Efficient Light Field Transmission via Enhanced Resampling Reconstruction and User Angular Attention Estimation", under review.

### Requirements

matplotlib==3.3.0, numpy==1.23.5, pandas==1.0.5, Pillow==9.5.0, scipy==1.10.1, seaborn==0.10.1, tensorflow==2.10.1, opencv-python==4.9.0.80


## How to Predict User Angular Attention? 

1. Place the light field image (LFI) (in form of subviews) into the folder: `./Dataset/APP/MINI_BATCH_DATA_TEST`. For convenience, a sample LFI has already been placed in this folder.

2. Run `python3 app.py`. Note: If you are predicting LFI from a new dataset, ensure that your LFI have a resolution of `9 × 9 × 600 × 400 × 3` (either cropping or up/downsampling).

3. The results will be saved to: `./Datasets/quality_predictions`.

The predicted result is a probability distribution, representing the likelihood of each subview being viewed. The larger the value, the higher the probability of being viewed.





## Transmission Pipline

Since the proposed method uses HEVC codec, [ffmpeg](https://ffmpeg.org/download.html) is required for transmission.

### The Proposed Method (Angular Attention + Enhanced Resampling Reconstruction)

1. Specify the root directory of the light field images to be transmitted in `./Transmission/proposed.py`: `./Transmission/scenes/proposed`. For convenience, a sample LFI has already been placed in this folder.

2. Run `python3 ./Transmission/proposed.py`.

3. The transmitted Light Field Images (LFI) will be saved in `./Transmission/scenes/proposed/ComplexBackground/0062/0062_Divide_6`. Within this directory:

- `0062_Divide_6/Adaptive_SR/Reconstruction` contains the LFIs transmitted with the proposed strategy.
- `0062_Divide_6/Adaptive_SR/SR` contains the LFIs transmitted with the proposed strategy without residual map.
- `0062_Divide_6/SR/Reconstruction` contains the LFIs transmitted with the Enhanced Resampling Reconstructions-only method.
- `0062_Divide_6/SR/SR` contains the LFIs transmitted with the Enhanced Resampling Reconstructions-only method without residual information.



### Angular Attention-only strategy 

1. Specify the root directory of the LFI to be transmitted in `./Transmission/Angular_Attention_only.py`: ./Transmission/scenes/Angular_Attention_only` (recommended to use an absolute path).

2. Run `python3 ./Transmission/Angular_Attention_only.py`.

3. The LFI transmitted using the Angular Attention-only strategy will be saved in the `./Transmission/scenes/Angular_Attention_only/ComplexBackground/0062/0062_Divide_6/Adaptive/Adaptive_transmission` directory.



### HEVC

1. Specify the root directory of the LFI to be transmitted in `./Transmission/HEVC.py`: `./Transmission/scenes/HEVC`.

2. Run `python3 ./Transmission/HEVC.py`.

3. The LFI transmitted using the HEVC will be saved in the `./Transmission/scenes/HEVC/ComplexBackground/0062/0062_Divide_6/HEVC/HEVC_transmission` directory.



### JPEG2000

1. Specify the root directory of the LFI to be transmitted in `./Transmission/jpeg2000.py`: `./Transmission/scenes/JPEG2000`.

2. Run `python3 ./Transmission/jpeg2000.py`.

3. The LFI transmitted using the JPEG2000 will be saved in the `./Transmission/scenes/JPEG2000/ComplexBackground/0062/decompressed` directory.



