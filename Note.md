# SRN-DeblurNet Note

## 20

steps: 127000
Generator - Norm: Instance
cosine restart: t_mul=2.0, m_mul=1.0, alpha=0.1
exponential decay: 0.999^(step/1000)

## 21

steps: 127000
cosine restart: t_mul=2.0, m_mul=0.9, alpha=0.1
exponential decay: 0.997^(step/1000)
forward: ~125ms

## 22

Generator - Norm: Batch
(FusedBatchNorm faster than InstanceNorm but slightly worse)
forward: ~90ms

## 23

Generator - Norm: None

## 24

(unchanged)
Activation: Swish => ReLU

## 25

lr: 5e-4

## 26

(unchanged)
remove SEUnit in ResBlock
lr: 5e-4

## 27

lr: 1e-3
same as ##23

## 28

replace L2 regularizer + Adam with AdamW
lr: 1e-3
wd: 1e-4

## 29~34

steps: 31000
wd: 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5
(chose 5e-5)

## 33.2

steps: 127000

## 35,36,33,37,38,39

steps: 31000
lr: 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4
(chose 1e-3)

## 40,33,41,42

batch size: 48,32,20,16
(chose 32)

## 43

(unchanged)
steps: 255000
batch size: 16

## 50

steps: 255000
batch size : 32
Update train set: Pixiv bookmark

## 51

steps: 511000
Update train set: Pixiv bookmark + MSCOCO 2017 + DIV2K splitted + Flickr2K splitted, with filtering

## 52

(unchanged)
steps: 511000
remove SEUnit in ResBlock

## 53

(unchanged)
steps: 511000
remove SEUnit in ResBlock
downsample: strided convolution => SpaceToDepth

## 54

steps: 127000
model: GeneratorSRN
added end-to-end skip connection
forward: 73.669ms

## 55

steps: 127000
model: GeneratorResUNet
forward: 63.704ms

## 56

steps: 127000
model: GeneratorResNet
forward: 51.880ms

## 57

steps: 127000
model: GeneratorResNet
added 1 ResBlock before ResBlocks and 1 ResBlock after ResBlocks, with U-Net-like skip connection
forward: 64.402ms

