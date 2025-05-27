# RAVEN: Official Repository of Real-Time Audio-Visual Speech Enhancement Using Pre-trained Visual Representations
This is the official repository of the paper **R**eal-Time **A**udio-**V**isual Speech **En**hancement Using Pre-trained Visual Representations, accepted at Interspeech 2025.


## Usage Instruction
Clone this GitHub repo and run 
```
git submodule update --init --recursive
```


Create a virtual environment:
```bash
conda create -y -n avse python=3.8
conda activate avse
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

:exclamation: FIRST, change `PROJECT_ROOT_PATH` in `config.py` before proceeding.

Then run 
`export PYTHONPATH='/your/path/to/this_project'` in your terminal and change directory to `src` folder. 



## Data Preprocessing

We use [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) for our training. Please download the dataset and change the `DATA_FOLDER_PATH` in `config.py` to the folder path you saved the data to. 

We also use [MUSAN](https://www.openslr.org/17/) for creating our noisy input mixture. Please download the dataset and change the `MUSAN_FOLDER_PATH` to the folder path you saved the MUSAN data to. 

You can find our train/val/test split of VoxCeleb2 at `src/data/split.parquet`, and the train/val/test split of MUSAN at `src/data/musan_split.csv`. 

### Extract Pretrained Visual Embeddings

Prior to extracting the pretrained embeddings of the dataset, clone the corresponding GitHub repo into the project root folder and download the checkpoints of the selected pretrained model. Follow the environment setup instructions in each pretrained model's Github README and then run the feature extractor script in the setup environment from `src` folder.

| Encoder Task | Encoder Name | GitHub Repo | Checkpoint Used | Feature Extractor Script |
|--------------|--------------|-------------|-----------------|---------------------------|
| AVSR         | [VSRiW](https://arxiv.org/pdf/2202.13084)        |[link](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages): save to `/benchmarks/GRID/models/` | GRID visual-only unseen WER=4.8 [\[src\]](https://bit.ly/3patMVh)   | `src/data/VSRiW_extract_visual_features.py` |
| AVSR         | AVHuBERT [Paper 1](https://arxiv.sorg/abs/2201.02184), [2](https://arxiv.org/pdf/2201.01763)    |[link](https://github.com/facebookresearch/av_hubert)<sup>1</sup>: save to `avhubert/conf/finetune/`              |base fine-tuned for VSR on LRS3-433h [\[src\]](https://dl.fbaipublicfiles.com/avhubert/model/lrs3/vsr/base_lrs3_433h.pt)<sup>2</sup> |`src/data/avhubert_extract_visual_features.py` |
| ASD          | [TalkNet](https://arxiv.org/pdf/2107.06592) |[link](https://github.com/TaoRuijie/TalkNet-ASD): save to repo root folder |  [\[src\]](https://drive.google.com/file/d/1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm/view)   | `src/data/TalkNet_extract_visual_features.py`|
| ASD          | [LoCoNet](https://arxiv.org/abs/2301.08237)      | [link](https://github.com/SJTUwxz/LoCoNet_ASD): save to repo root folder |[\[src\]](https://drive.google.com/file/d/1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK/view) |`src/data/LoCoNet_extract_visual_features.py` |


<sup>1</sup> Follow instructions in the Github repo README then downgrade `omegaconf==2.0.1` and `hydra-core==1.0.0`; you need `pip < 2.4.0` to install `omegaconf==2.0.1` and you may also need `numpy < 1.24`

<sup>2</sup> Go to the official model checkpoint [page](https://facebookresearch.github.io/av_hubert/) and sign the license agreement first. 


### Simulate the Noisy Input Mixture

For faster training and processing, convert all the m4a's to wav's. Run the below command to create the noisy input mixture, assuming you are at `src` level.
```
python -W ignore utils/mix_speech_gpu.py
```

## Training
Run the below terminal command to start training the model. By default, logs and checkpoints will be saved to the CHECKPOINT_DIR defined in config.py. You can override parameters like visual encoder, batch size, and checkpoint directory using command-line arguments.

```
python -W ignore train.py
```
To resume training from a saved checkpoint, add the --train_from_checkpoint flag and specify the path using --ckpt_path:

```
python -W ignore train.py --train_from_checkpoint --ckpt_path=checkpoints/epoch-last.ckpt
```

## Evaluation

First generate test input mixtures of different conditions and SNR scenarios, run 
```
python -W ignore data/generate_test_data.py --condition=noise_only --snr=-10
```
If you want to generate different conditions and snrs at once, you could use comma to separate them. For example: `--condition="noise_only, one_interfering_speaker, three_interfering_speakers" --snr="mixed, -10, -5, 0"`

The mixture will be saved to `/path/to/VoxCeleb2/dev/mixed_wav/{condition}/{snr}/`. 


To evaluate a single visual encoder under a specific test condition and SNR, use test.py, which accepts command-line arguments for full flexibility:

```
python -W ignore test.py --visual_encoder=TalkNet --test_condition=noise_only --test_snr=-10
```




