# XPL: A Cross-Model framework for Semi-Supervised Prompt Learning in Vision-Language Models

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

We provide the running scripts in `scripts/`

Below we provide examples on how to run XPL experiments on `Eurosat`.

**Ours (Cross-model multi-modal CoOp w/ unlbl data)**:

- 5% : `CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/xpl.sh eurosat vit_b16 end 16 5 False pt`
- 1 shot: `CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/xpl.sh eurosat vit_b16 end 16 1 False shot`


## How to get the average of 3 seeds

- Sample code for getting the 3 seeds average of `CoOp w/ visual and text prompt only` for `Eurosat 10%`

`python parse_test_res.py ./output_XPL/eurosat/vit_b16_1pt`

Here `./output_XPL/eurosat/vit_b16_1pt` represents the path to the corresponding `seed1`, `seed2` and `seed3` folder.