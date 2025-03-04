# Generative AI with Large Language Models (LLM)

- **PREREQS: Basic data science and Python**
- **Course Objectives**: explore steps in generative ai lifecycle: scoping, select model, optimize model for deployment, and integrate into application. Specifically around LLM (so text primarly)
- [FAQ for GenAI with LLMs Labs](https://community.deeplearning.ai/t/genai-with-llms-lab-faq/374869)
- If you cannot find an answer in the FAQs, you can [search for or create a new post here](https://community.deeplearning.ai/c/course-q-a/generative-ai-with-large-language-models/328)

---

## Overview

- **Week 1** - transformer, training models, compute resources, _in-context learning_, how to tune the most important generation params of output
- **Week 2** - adapt pretrained models for needs using _instruction fine tuning_
- **Week 3** - align output of models with _human values_ to increase helpfulness and decrease harm and toxicity
- Work done in **AWS** at _NO COST to you_
- General purpose technologies - good for mulitple applications
- Lots of important work over many years to identify use cases / fine tune / discovery work
- Very few know how to build applications using OMs
- Lots of input from other companies
- AI enthusiasts, engineers, and data scientists looking to to learn technical foundations of how LLMs work
- Training/tuning/deploying LLM

---

## Contributors

- Ehsan Kamalinejad, Ph.D. - Machine Learning Applied Scientist, AWS
- Nashlie Sephus, Ph.D. - Principal Technology Evangelist for Amazon AI, AWS
- Saleh Soltan, Ph.D. - Senior Applied Scientist, Amazon Alexa
- Heiko Hotz - Senior Solutions Architect for AI & Machine Learning, AWS
- Philipp Schmid - Technical Lead, Hugging Face and AWS ML Hero

---

## Generative AI & LLMs

- A machine capable of making content that mimics human capability
- Machines find _statistical patterns_ in tons of datasets created by humans
- Foundational / Base LLMs
  - Larger parameters: GPT, BLOOM (176billion), _FLAN-T5_ (will use this one)
  - Smaller: LLaMa, PaLM, BERT (110million params in Bert-base)
  - **Parameters** - think of it as MEMORY and the larger the parameters the larger the memory/ability to figure things out
- Syntax/Writing code (_typical programming_) vs _Prompt Programming_

---

### Prompt -> Inference -> Completion

- You **PROMPT** a model with a question
- **INFERENCE** is the act of using a model to generate text
- Model output (**COMPLETION**) _predicts the next words_ and creates output including the original prompt and the answer.

---

## Use Cases

- Chatbot, but this is overused as an example
- Write an essay
- Translate languages
- Translate natural language to programming code
- Identify people/places in an article
- Connect to external APIs and then process that information to respond (i.e. check a flight)

---

## Text Generation Before Transformers

- RNN (recurrent neural networks) were used for text generation in the past
  - Used the previous few words to predict the text and would often get it wrong
- Many words are hominyms that have multiple meanings so **context** is so very important
- **2017 - _Attention Is All You Need_ (Univ Toronto and Google)** - Transformers arrived
  - Scale efficiently
  - Parallel process
  - Learn / pay attention to words/input

---

## Transformers Architecture

![General Architecture](architecture.png)

- **Transformer architecture**: Looking at really important parts to get the intuition you need to make practical use of these models
  - Make foundation around vision transformers
- Ability to learn the relevance and context of **all** of the words in a sentence. Not just to it's neighbor, but to other words in the sentence.
- This creates an _attention map_ is created between words. This is called `self-attention`
- 2 distinct parts: **encoder** and **decoder**
- Ultimately, these models are just _huge calculators_ and, as such, you must first **tokenize** the words and provide values for each.
- The Flow:

  1. Tokenize words in input and assign token ids to words/phrases

     - This tokenizer, once used here, **_MUST ALSO BE USED WHEN GENERATING TEXT_**

  2. Input now represented as numbers, passed to the **embedding** layer

     - Each token ID matched to a multi-dimensional vector where these vectors _learn to encode the meaning and context of individual tokens_ in the input sequence.
     - Ok if not familiar with **multi-dimensional vector learning**
     - Once each item is assigned a vector, they can be mapped out into space where then the angle can be calculated and mathematically be understood.
     - _Positional encoding_ also occurs to perserve the position of each token in the input sequence (i.e. word order)

  3. Vectors are passed to the `self-attention` layer of the **encoder** allowing the model to analyze the relationships between tokens in the input sequence.

     - It is at this point that _the self-attention weights learned during training and stored in these layers_ can be used to determine the importance of each word in the input sequence to other words in the sequence.
     - The _Transformer actually has **multi-headed** self attention_ allowing it to learn independently of others in parallel. This can vary between 12 to 100 **heads**
       - Each **head** will learn a different aspect of language over time via training data
     - At the end of processing here, all attention weights have been applied to the input data.

  4. These weights are then passed to a **fully-cnnected feed-forward network**

     - Output from here is _a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary_
     - These _logits_ can then be passed to a final **softmax** layer, where they are notmalized into a probability score for each word. It includes a probability for _EVERY SINGLE WORD IN THE VOCABULARY_ so there are thousands of scores here.
     - One token will have a score higher than the rest, that that **is the most likely predicted token**

  5. Data that leaves the **encoder** is a _deep representation of the structure and meaning of the input sentence_ and is inserted into the _middle_ of the **decoder** to _influence_ the decoder's self-attention mechanisms.

---

## Generating Text with Transformers

- Walk through a translation task or a _sequence-to-sequence_ task (which was the original bojective of the transformer architecture designers)
- Translate French to English
- Data that leaves the encoder goes to the _middle_ of the decoder to _influence_ it's self-attention.
- A _start of sequence token_ is added to the input to the decoder, triggering the decoder to predict the next token, based on it's contextual understanding provided by the encoder.
- After the `softmax` output layer, we have our **first token**
- This then loops back to the **decoder** passing in the output token, _triggering the generation of the next token_ until the model predicts an _end-of-sequence_ token.
- Tokens are then _detokenized_ into words, and you have the output
- Output from `softmax` layer can be used in many ways to influence how creative your generated text is.

---

### Encoder / Decoder Model Types

> In general, the larger the model, the more capable the model will be in carrying out it's tasks. There could be a new Moore's Law for LLMs.

- Training larger and larger models because infeasible and very expensive
- The **encoder** and **decoder** can be separated out into different spaces
  - Also called **autoencoding** models
  - Build _bidirectional_ representations of the input sequence
  - **Pre-Training Strategy:** `MLM | Masked Language Modeling`
  - **Use Cases:** Sentiment analysis, named entity recognition, word classification
  - **Examples:** BERT and ROBERTA
- BART and T5 have _both_ **encoder** and **decoder**
  - Also called **sequence-to-sequence** models
  - Span corruption is used to train
    - Masked inputs replaced by a `Sentinel token`
  - The decoder is then tasks with replacing the `sentinel token` with the correct token
  - **Pre-Training Strategy:** `Span Corruption`
  - **Use Cases:** translation, text summarization, question answering
  - **Examples:** BART and T5
- **DECODER-ONLY** models are most commonly used today - GPT family, BLOOM, Jurassic, LLaMA and many more. Can be applied to most tasks.
  - Also called **autoregressive** models
  - `pre-trained` with causal data
  - Mask the input sequence and can only see the tokens **UP TO** the token
  - UNIDIRECTIONAL - must build up knowledge by a massive amount of examples as they have no context AFTER the token.
  - **Pre-Training Strategy:** `CLM | Causal Language Modeling`
  - **Use Cases:** Text Generation
  - **Examples:** GPT and Bloom

---

## Prompt Engineering

- **Prompt** - text fed into model
- **Inference** - act of generating text
- **Completion** - output from model
- **Context Window** - memory for the model
- **Prompt Engineering** - to develop and improve the prompt

  - Remember your **Context Window**!!

    - You have a _limit_ on the amount of _in-contenxt learning_ you can pass into the model
    - Smaller models are good at a small number of tasks
    - Larger models can be good at things that they weren't even trained to do.

  - **In-Context Learning** - Provide EXAMPLES or additional details inside the context window

    - **Zero-Shot Inference**

      - _Example_:

        ```
        Classify this review:
        I loved this movie!
        Sentiment:
        ```

    - **One-Shot Inference**

      - _Example_:

        ```
        Classify this review:
        I loved this movie!
        Sentiment: Positive

        Classify this review:
        I don't like this chair.
        Sentiment:
        ```

    - **Few-Shot Inference**

      - Above 5 or 6 just doesn't help the model after that

      - _Example_:

        ```
        Classify this review:
        I loved this movie!
        Sentiment: Positive

        Classify this review:
        I don't like this chair.
        Sentiment: Negative

        Classify this review:
        This is not great.
        Sentiment:
        ```

---

## Generative Configuration

- **Inference parameters** - exposed set of configuration parameters that can be set for model
  - Different than _training_ and tuning
- **Max new token** - the maximum number the process will go through the selection process. This can end early as the model might generate a stop condition before reaching the max
- Sampling Types
  - Most of the time, models use **greedy** decoding, using the _highest propability_ tokens
  - You can also use **random** sampling to select an output word at random using the probably distribution to weight the selection. The following are used to _limit_ the sampling so that the responses are reasonable.
    - **Sample top K** - top tokens - can limit the top-k results with highest propability (i.e. 3 would be top 3 based on probability rating)
    - **Sample top P** - top cumulative probability -limit to the top results with a _cumulative probability_ of p (i.e. 0.3 would be top results with total summed probability of less than 0.3)
- **Temperature** - modifies the probability distribution of the _NEXT_ token.
  - Low temperature, more strongly peaked with the higher probability in the middle of the
  - High temperature, more varability, broader flatter distribution
  - A temperature of 0.1 will often result in the same response over and over

---

## Generative AI Project Lifecycle

![Generative AI Project Lifecycle](generative-ai-lifecycle.png)

Be _specific_ to save time and compute costs.

1. **SCOPE** - Define use case as _narrowly_ as you can
2. **SELECT** - Decide if using a foundational model _off the shelf_ or if you're _pre-training_ your own model (type of model, size of model 100 billion or bigger VS 1 to 30 billion VS sub-billion parameter model)
   - Big difference between google AI that needs to know everything VS a chat bot for a single small company
3. **ADAPT AND ALIGN MODEL**
   - Prompt Engineering
   - Fine-tuning
   - Align with Human Feedback (additional fine tuning) - ensure your model behaves well
   - Evaluate model
4. **APPLICATION INTEGRATION**
   - Optimize and deploy model for inference
   - Augment model and build _LLM-powered applications_

---

## Lab Instructions

- You will have 2 hours to complete each lab
- There is a green button on the Coursera slide that will take you to AWS (you must NOT be signed in when you click)
- Once redirected, you'll want to search for `Sagemaker`. Follow the steps from the `Vacareum` page after launching the lab

---

## Pre-training Large Language Models

- Model hubs - have model cards that describe how they work and how they are better at specific tasks
- Deep statistical understanding of language based on `pre-training`
- If your language is very specific or not used in the real world (such as legal or medical) you may need to do `pre-training` from scratch
- GB/TB/PB of unstructured data that the model processes
- Encoder generates the vector representation for each token
- Need to process data scraped to remove bad data

---

## Computational Challenges of Training LLMs

### Memory

| Consideration                          | Bytes / Parameter | For 4 Billion Params |
| -------------------------------------- | ----------------- | -------------------- |
| 1 parameter = 32-bit float             | 4 bytes           | 4gb @ 32-bit         |
| Adam Optimizer (2 states)              | +8 bytes          | +8gb @ 32-bit        |
| Gradients                              | +4 bytes          | +4gb @32-bit         |
| Activations and Temp Memory (Var size) | +8 bytes          | +8gb @ 32-bit        |

- **Memory** is definitely the number one challenge.
  - For every parameter, you'll need **6x** the memory for all the `pre-training`
  - A 1 billion parameters, it's 4gb of memory just for parameters, and 24gb for all memory needed to train
- Ways to reduce costs/necessary memory?
  - **Quantization:** Can use 16-bit Floating Point | 8-bit Integer (2 bytes memory) instead of the 32-bit floating points (4 bytes memory)
    - **BFLOAT16:** _NEW!_ Significantly helps with training stability by splitting bits between exponent and fraction
      - Can represent 32-bit floating numbers in lower-precision

| Type     | Bits | Sign | Exponent | Fraction | Memory / Val | Storage of 1B Params |
| -------- | ---- | ---- | -------- | -------- | ------------ | -------------------- |
| FP32     | 32   | 1    | 8        | 23       | 4 bytes      | 4gb                  |
| FP16     | 16   | 1    | 5        | 10       | 2 bytes      | 2gb                  |
| BFLOAT16 | 16   | 1    | 8        | 7        | 2 bytes      | 2gb                  |
| INT8     | 8    | 1    | 0        | 7        | 1 byte       | 1gb                  |

- _**NOTE:**_ The model can still **store** 1 billion parameters regardless of which type you choose, they just take up a lot less space
- Some newer models support up to **500 BILLION PARAMETERS** requiring 500 times the storage laid out above.
- Impossible to do on a single GPU...would need multiple GPUs to process in parallel
- **PyTorch** can handle the distribution of parameters to multiple GPUs. There are a few approaches that it supports:
  - **Distributed Data Parallel (DDP)** each GPU requires a full copy of all parameters, but only processes a specific portion of the data
  - **Fully Sharded Data Parallel (FSDP)** ZeRO redundancy uses 3 levels of _sharding_ to reduce redundancy of data and therefore memory needs, BUT increases communication between GPUs
    - _Pos_ - only optimizer states are shareded
    - _Pos+g_ - optimizer states AND gradients are sharded
    - _Pos+g+p_ - optimizer states, gradients, and parameters are sharded (sharding across 64 GPUs would reduce size by a factor of 64)
  - Performance (DDP) vs Memory (FSDP) tradeoff here
  - FSDP can drop the number of TeraFLOPS (1 trillion floating point operations / second) per GPU dramatically when using sharding (FSDP)

## Scaling Choices

- Smaller models
- Smaller dataset size
- Compute budget (GPUs, training time)
  - If your compute budget is _FIXED_, you're only option is to modify data size and model size.
- 1 petaflop/s-day
  - 1 quadrillion floating operations per second
  - _Examples_
    - 8 NVIDIA V100 GPUs operating a full efficiency for an entire day
    - 2 NVIDIA A100 GPUs (equivalent to the 8 above)
- **T5 XL 3 billion parameters - required about 100 petaflop/s-day**
- Chincilla Paper - argues that models are **over-parametertized** and **under-trained**
  - i.e. don't need as many params, just more training data
