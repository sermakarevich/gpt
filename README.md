It’s always a pity when you finish an interesting book. Especially when it is a series. Now a new search is ahead which would be made of trials and errors. But what if you can write a continuation of the book ? Not on your own, of course, but with a neural network.  Andrej Karpathy has an awesome series of videos on his youtube channel, explaining neural network concepts. In one of the videos he explained how the GPT model works and I decided to try it out on Liu Cixin trilogy called “The Three body problem”. 

Andrej explains the GPT concept in progressive manner. He starts from the very simple bi-gram model and improves it by explaining and adding small pieces. We will do the same, I kept all major models implementations so you can try each of them and compare the differences. There will be 10 model implementations. Most of the models you can train with just a CPU, but if you would like the model to grow bigger and perform better, a GPU might be beneficial. The biggest model I trained during experimentation consumed 5GB of GPU memory and training for 1K epochs out of 10K epochs required, took around 7 minutes. So on CPU this model should train overnight just fine.  

Lets list resources here:
- Andrej Karpathy video lecture: https://youtu.be/kCc8FmEb1nY
- github: https://github.com/sermakarevich/gpt/ 
- Original paper: Attention is all you need https://arxiv.org/abs/1706.03762 

### Data
I used a Liu Cixin “The three body problem” trilogy, concatenated three books and put it into `/data/data.txt` file. Feel free to try your favourite book. 

Text has to be transformed into tokens which we later substitute with embeddings - numeric vector representation which values are optimised during the training process. We will use the simplest tokenisation possible - a single character will be our token. Feel free to check more advanced tokenizers used by Google and OpenAi:
https://github.com/google/sentencepiece
https://github.com/openai/tiktoken
They use tens of thousands of tokens while in my case I got only around 150 characters. 

Every character is mapped to an integer value which would mean a position in the embedding layer. Reverse mapping is required to decode model outputs which would be a sequence of integers where max_value is a vocabulary size. 

We split the dataset into train and validation sets. First is used to train a model and second one is used to check how well our model performs on data it hasn’t seen before. I used a 90/10 split. 

To submit data into the model we will use a simple function which generates random indices within train/val dataset ranges, stack them together to get a batch of block_size chars. block_size is the number of tokens in a sequence we would like to show our model.  

### Models
#### BigramLanguageModelV1
Model consists of only the Embedding layer, which substitutes token indices with vector representation of n_embed length. If we use n_embed = vocab_size that would allow us to treat embeddings as logits. Training this model means learning distribution of the next character in the text having only the current single character. 

#### BigramLanguageModelV2
If we would like the Embedding layer to have n_embed dimension different than vocab_size, we would need to map that output of n_embed into vocab_size to get logits. Thus we add additional init model input, n_embed and additional linear Layer. 

#### BigramLanguageModelV3
Implements positional awareness through an additional Embedding layer. Each token now gets an additional embedding vector which corresponds to its position in a sequence. Both embedding vectors are summed up before passing into a fully connected (linear) layer.

#### BigramLanguageModelV4
Attention is a key concept in neural networks these days. SAM/CLIP/BLIP models which do image segmentation / image-text similarity / image captioning - all based on a similar architecture which uses Attention mechanism. In a few words - every token submits three vectors called key/query/value which are transposed and multiplied in a specific way, so that a token at position T can get access to values of all tokens at positions [0:T-1]. This helps the token to understand its position and surrounding context better.  To prevent context leakage, future tokens at [T+1:] are hidden by zeroing out the upper triangular part of the weights = torch.matmul(query, key.transpose(1, 2)).  

#### BigramLanguageModelV5
One self-attention layer is good but multiple is better. We create num_heads which controls how many attention layers we would like to have. Outputs from heads are concatenated, so to get back to the original size, the attention layer should map input_size into input_size / num_heads. Additional fully connected layer is applied on top of concatenated outputs from multiple heads. 

#### BigramLanguageModelV6
We improve the model with a feed forward part which consists of a sequential Linear layer and RELU activation. We use it to multiply the output of the self attention part.  

#### BigramLanguageModelV7
We start to scale our model. New class is called Block which contains multi head self attention followed by a feed forward network. We use not a single block now, but multiple, chained together sequentially - so that output from the previous Block is the input into the next Block. 

#### BigramLanguageModelV8
Implements skip-connection. The idea is taken from this paper https://arxiv.org/abs/1512.03385 and is quite simple. To fight the vanishing gradients problem, output of a part of the model is summed up with the input. According to the calculus chain rule, the gradient is split between two parts equally in case of sum operation. Thus helps to not decrease gradient too quickly when the model grows bigger. 

#### BigramLanguageModelV9
LayerNormalisation helps to control distribution of weights. Its description can be found in the paper https://arxiv.org/pdf/1607.06450.pdf. We add it after the sequence of Blocks and inside Blocks before passing input into self attention mechanism and before passing output from self attention into the feed forward network part. 

Feed forward part got upgraded with additional linear layer after RELU and fan-out / fan-in: 
first linear layer maps input of size in_size into out_size = in_size * 4
second layers maps out_size back to in_size

#### BigramLanguageModelV10
Now we want to scale what we have implemented so far to the limit:
- when model gets bigger we need to control it for overfitting by adding Dropouts to 
  - self attention
  - multi head
  - feed forward part
- we implement control of the number of blocks in sequence with block_layers param.
- we increase our parameters to make our model bigger:
  - n_embed
  - num_heads
  - block_size
  - block_layers

### Training
A trainer.py script can be found in the repo, which you can just run. It would iterate over all 10 models implemented, train them for epochs, iterations, print loss and generate generate_chars tokens.  

In my case loss went 
- from: 2.5 BigramLanguageModelV1
- to: 1.1 BigramLanguageModelV10

### Text examples:
#### BigramLanguageModelV1

> I’tansey, pstheche TONAncoundor nst sethentte torepounes re e praī; evive pe aö; fly y orw sentoull, t. ce wans a hed hese c[ongind onilrysphan be us aro t ayole s tathein ficad tofe. ivethe outonly epin tem: t bureionars, »Linod rusosthinfikin I g turoo owearro bl imlee “Mo thepiomo, as a A tyes rdinfine, tigh’sherd w tare oblle teng we andan’t. t in penoughe ave ppoGathy t h ot f tonting roe t f he?”

#### BigramLanguageModelV10

> “People willing to keep my news: To a public retiring at the doom thing.”
Most realizationist. “We’ll met even the Wallfacers’ own of the support caw of happy. The Trisolar invaders were resorteous.… We’ve just confirmed the calcular: It’s not too located establishmentees, and good. It’s offering to paradise for if he speak to superior than lots. For now, I asked the employon all altogether the ships.”
“Of course, to should we care for another specimen.…”

Model output is non-deterministic since in generating we sample from probabilities. If you would like to make it deterministic - you would need to get argmax from probabilities. 

It does not look like  Liu Cixin just yet, but we obviously improved in version 10 compared to version 1. The biggest limitation atm is the tokenization - we train model to predict a single character which is, probably, harder than predicting tokens which are made of multiple characters or the whole words. In this case, however, our model should be much bigger. Embedding layers would grow from 150 indices to 50K fe. 
