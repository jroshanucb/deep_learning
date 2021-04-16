Auto Regressive (AR) models use tokens generated in previous time-step as input to calculate outputs. The Transformers are by-and-large autoregressive in nature. Coupled with an attention mechanism, an auto regressive transformer model gets the right context thereby increasing its accuracy. In contrast, Non-Auto Regressive (NAR) models generate a sequence of tokens in parallel removing the reliance on the tokens from previous time-steps. This approach significantly reduces the inference latency of the output. However, at the expense of low accuracy. In this project, we explore the implementations of both these approaches and assess options to narrow the accuracy gap. We changed the architecture of the decoder as part of the NAR implementation and introduced Conditional Random Fields (CRF) to generate the output sequence of a machine translation task. We used IWSLT dataset for German (DE) to English (EN) translation task. We were able to train both the models from scratch on a GPU based server and we observed BLEU scores of AR & NAR to be 16.07 and 8.79 respectively.

The AR model is implemented in std_transformer.py

To run this code, ensure torchtext installed is set of version 0.6.0 (pip install torchtext==0.6.0)

or change the import as follows to use the latest version of torchtext (0.9.0)

from torchtext.legacy.datasets import IWSLT, Multi30k

setup python version to either 3.6.x or 3.8.x

"python std_transformer.py" to train the model

Training on https://gpu.land/ using a single GPU (Tesla V100) provides the best performance. 1000+ epochs can be run within 24 hours.

