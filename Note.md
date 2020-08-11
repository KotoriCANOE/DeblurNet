# SRN-DeblurNet Note

---

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

## 58

steps: 127000
model: GeneratorResUNet
2 EBlocks + 2 DBlocks
forward: 63.676ms

## 59

(unchanged)
steps: 127000
model: GeneratorResUNet
6 ResBlocks + 6 ResBlocks
forward: 78.625ms

## 60

steps: 127000
model: GeneratorSRN
loss: L1 + MS-SSIM2
forward: 83.02ms

## 61

steps: 511000
model: GeneratorSRN
loss: L1 + MS-SSIM2
exponential decay: 0.997 => 0.998
forward: 76.151ms

## 62

steps: 1023000
MS-SSIM2: [0.5, 1.5, 4.0] => [1.5, 4.0, 10.0]
forward: 78.232ms

## 63

(unchanged)
steps: 260000 (250000)
Activation: Swish => ReLU

## 64

steps: 274000 (250000)
Remove biases in ResBlocks

## 65

steps: 205000
Remove biases in EBlocks and DBlocks
forward: 70.891ms

## 66

(unchanged)
steps: 2047000
exponential decay: 0.998 => 0.999
Activation: Swish => ReLU
forward: 64.576ms

## 67

steps: 2047000
exponential decay: 0.998 => 0.999
Activation: Swish

## 68

(unchanged)
steps: 2047000
exponential decay: 0.998

## 69

(unchanged, failed due to LR policy)
steps: 2047000
Added one more block for encoder/decoder

## 70

(unchanged)
steps: 511000 (470000)
exponential decay: 0.998
SEUnit: off
loss: L1

## 71

(optional)
steps: 2047000
exponential decay: 0.999
SEUnit: off
loss: L1

## 72

steps: 255000
(same as ##67) channels: [32, 32, 64, 96, 128]
Added warm up for LR schedule (63000 steps)

## 73

(unchanged)
steps: 255000
InBlock & DBlock_0: 32c => 64c

## 74

(unchanged)
steps: 255000
channels: [32, 16, 32, 64, 128]

## 75

(unchanged)
steps: 255000
channels: [48, 24, 48, 96, 192]

## 76

steps: 1023000
(same as ##67) ResBlocks: 0, 2, 2, 2, 2, 2, 2, 2, 2
Anti-aliasing re-sampling with [1, 2, 1] filter

## 77

(unchanged)
steps: 1023000
Removed EBlock_0 and DBlock_0
ResBlocks: 0, 1, 2, 8, 2, 1, 0
LR: 5e-4

## 78

(unchanged)
steps: 390000/1023000
Removed EBlock_0 and DBlock_0
ResBlocks: 1, 2, 2, 4, 2, 2, 1
LR: 7e-4

## 79

(chosen)
steps: 1023000, 2047000
(same as ##67) ResBlocks: 0, 2, 2, 2, 2, 2, 2, 2, 2
Anti-aliasing re-sampling without [1, 2, 1] filter

## 80

use a better LR warmup (first linear, then cosine restart, multiply by exponential decay)
(unchanged)
steps: 1023000
Removed EBlock_0 and DBlock_0
ResBlocks: 2, 2, 2, 2, 2, 2, 2
LR: 5e-4

## 81

(unchanged)
steps: 1023000
DBlocks: ResBlocks last => ResBlocks first
LR: 7e-4

---

VDSR (20 conv layers)

## 100

steps: 255000
activation: ReLU
cosine restarts (warmup_cycle=6)
exp decay: 0.998
LR: 1e-3

## 101, 102

steps: 255000
activation: ReLU
cosine restarts (warmup_cycle=0)
exp decay: 0.998
LR: 1e-3

## 103

steps: 255000
activation: Swish
cosine restarts (warmup_cycle=0)
exp decay: 0.998
LR: 1e-3

## 104

steps: 255000
activation: ReLU
cosine restarts (warmup_cycle=0)
exp decay: 0.998
LR: 1e-3
grad clip: 0.05

## 105

steps: 255000
activation: ReLU
cosine restarts (warmup_cycle=6)
exp decay: 0.998
LR: 1e-2
grad clip: 0.05

## 106

steps: 255000
activation: ReLU
cosine restarts (warmup_cycle=6)
exp decay: 0.998
LR: 2e-3
grad clip: 0.1

## 107

steps: 255000
activation: ReLU
cosine restarts (warmup_cycle=6)
exp decay: 0.998
LR: 1e-3
grad clip: 0.1

---

SRN

## 108

(good)
steps: 511000
cosine restarts (warmup_cycle=6)
exp decay: 0.998
LR: 1e-3
grad clip: 0.1

## 109

(good)
steps: 511000
LR: 1e-3
grad clip: 0.2

## 110

steps: 511000
LR: 1e-3
grad clip: 0.05

## 111

(unstable)
steps: 511000
LR: 1e-3
grad clip: 0

## 112

(failed)
steps: 511000
LR: 2e-3
grad clip: 0.1

## 113

steps: 511000
LR: 1e-3
grad clip: -0.4
(soft clipping)

## 114

(chosen)
steps: 511000
LR: 1e-3
grad clip: -0.2

## 115

steps: 511000
LR: 1e-3
grad clip: -0.1

---

## 120

steps: 255000
cosine restarts (warmup cycle=5)
dataset: Pixiv bookmark (epoch=1)
size: 8x3x256x256
new pre-processing

## 121

steps: 511000
cosine restarts (warmup cycle=6)
dataset: Pixiv bookmark (epoch=1)

## 122

steps: 511000
dataset: Pixiv bookmark (epoch=1)
improved pre-processing
val size: 256 => 64
(use packed data but forgot to set batch size, so val size is unexpected)

## 123

steps: 511000
dataset: Pixiv bookmark (epoch=2)

## 124

steps: 511000
dataset: Pixiv bookmark (epoch=2)
random seed: 0 => None

## 125

steps: 511000
dataset: Pixiv bookmark (epoch=2)
random shuffle before selecting val set

## 126

(test loss raises at the end)
steps: 511000
dataset: Pixiv bookmark + COCO (epoch=4)
size: 32x3x128x128
val size: 64 => 256
random seed: 0

## 127

(slight improvement)
steps: 511000
dataset: Mixup (epoch=1)

## 128

steps: 511000
dataset: Mixup (epoch=4)

## 129

(worse than ## 126, especially on linear loss)
steps: 511000
dataset: no Mixup (epoch=4)
tanh: on
(add tanh to the output)

## 130

(improve quite a lot)
steps: 511000
dataset: no Mixup (epoch=4)
predict in linear light (BT.709), loss in linear light
loss: BT.709 transfer

## 131

(slightly better than ## 130 regarding loss in gamma)
steps: 511000
dataset: no Mixup (epoch=4)
predict in linear light (BT.709), loss in gamma corrected
loss: BT.709 transfer
(add epsilon to avoid inf gradients)

## 132

(worse than ## 126)
steps: 255000
dataset: BT.709 to linear (epoch=1)
Mixup: on - linear on inputs, linear on labels
predict in linear light (BT.709), loss in gamma corrected
loss: BT.709 transfer
(wrong implementation of Mixup, where the lambda within each batch are the same)

## 133

(worse than ## 126)
steps: 127000
dataset: BT.709 to linear (epoch=5)
Mixup: on - linear on inputs, linear on labels
loss: BT.709 transfer
(wrong implementation of Mixup, where the lambda within each batch are the same)

## 134

steps: 511000
dataset: BT.709 to linear (epoch=8)
Mixup: off
loss: BT.709 transfer

## 135

steps: 511000
dataset: BT.709 to linear (epoch=8)
Mixup: on - linear on inputs, linear on labels
loss: BT.709 transfer
(fix implementation of Mixup)

## 136

steps: 255000
dataset: sRGB to linear (epoch=2)
Mixup: off
loss: sRGB transfer
(ZIMG implement BT.709 as a gamma=2.4 curve, so change to sRGB)

## 137

(worse than ## 136)
steps: 255000
dataset: sRGB to linear (epoch=2)
Mixup: on - linear on inputs, linear on labels
loss: sRGB transfer

## 138

(worse than ## 136)
steps: 255000
dataset: sRGB to linear (epoch=2)
Mixup: on - linear on inputs, linear on labels
loss: linear transfer

## 139

steps: 255000|511000
dataset: sRGB to linear (epoch=2|8)
Mixup: off
loss: sRGB transfer
tanh: off
(remove tanh)

## 140

(worse)
steps: 255000
dataset: sRGB to linear (epoch=2)
Mixup: on - linear on inputs, gamma on labels
loss: sRGB transfer
tanh: off

## 141

steps: 255000|511000
dataset: sRGB to linear (epoch=2|8)
Mixup: off
loss: linear transfer
tanh: off

## 142

steps: 255000|511000
dataset: sRGB to linear (epoch=2|8)
Mixup: on - linear on inputs, linear on outputs
loss: sRGB transfer
tanh: off

## 143

steps: 255000|511000
dataset: sRGB to linear (epoch=2|8)
Mixup: on - linear on inputs, linear on outputs
loss: linear transfer
tanh: off

## 144

(best one in Mixup)
steps: 511000|2047000
dataset: sRGB to linear (epoch=8)
Mixup: on - sRGB on inputs, sRGB on outputs
loss: sRGB transfer

## 145

(chosen)
steps: 2047000
dataset: sRGB to linear (epoch=8)
Mixup: off
loss: sRGB transfer

## 146

steps: 2047000
dataset: sRGB to linear (epoch=8)
Mixup: off
loss: linear transfer

---

## 200

SR
steps: 255000
dataset: TrainPP/00 partial

## 201

steps: 63000
dataset: TrainPP/00

## 202

(unchanged)
steps: 63000
skip: add => concat

## 203

steps: 127000
dataset: TrainPP
exponential decay: 0.99
GeneratorSRN
SEUnit: off
loss: L1

## 204

(unchanged)
steps: 127000
GeneratorResNet

## 205

steps: 2047000
exponential decay: 0.999
GeneratorSRN
SEUnit: on
loss: L1 + MS-SSIM

