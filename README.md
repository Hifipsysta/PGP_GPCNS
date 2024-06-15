## DualPrompt-PGP + GPCNS

Then, install the packages below:

```
pytorch==1.13.1
torchvision==0.14.1
numpy==1.25.0
timm==0.6.7
sklearn==1.3.0
matplotlib
```

or you can install these packages with ```requirements.txt``` by: 

```
pip install -r requirements.txt
```

## Data preparation

If the datasets aren't ready, just run the training command and the datasets will be downloaded automatically in the `--data-path`.

## Training

To train a model via command line:

**For CIL (Class Incremental Learning) Settings:**

50-Split TinyImageNet

```
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        tinyimagenet_dualprompt_pgp \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --output_dir ./output \
        --epochs 5\
        --num_tasks 50
```

100-Split TinyImageNet

```
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        tinyimagenet_dualprompt_pgp \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --output_dir ./output \
        --epochs 5\
        --num_tasks 100
```

