# Generative AI with Large Language Models (LLM)

- **PREREQS: Basic data science and Python**
- **Course Objectives**: explore steps in generative ai lifecycle: scoping, select model, optimize model for deployment, and integrate into application. Specifically around LLM (so text primarly)
- [FAQ for GenAI with LLMs Labs](https://community.deeplearning.ai/t/genai-with-llms-lab-faq/374869)
- If you cannot find an answer in the FAQs, you can [search for or create a new post here](https://community.deeplearning.ai/c/course-q-a/generative-ai-with-large-language-models/328)
- [Lecture notes are available on the DeepLearning.AI Website](https://community.deeplearning.ai/t/genai-with-llms-lecture-notes/361913)

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

## Resources

### Generative AI Lifecycle

- [Generative AI on AWS: Building Context-Aware, Multimodal Reasoning Applications - This O'Reilly book dives deep into all phases of the generative AI lifecycle including model selection, fine-tuning, adapting, evaluation, deployment, and runtime optimizations.](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

### Transformer Architecture

- [Attention is All You Need - This paper introduced the Transformer architecture, with the core “self-attention” mechanism. This article was the foundation for LLMs.](https://arxiv.org/pdf/1706.03762)
- [BLOOM: BigScience 176B Model - BLOOM is a open-source LLM with 176B parameters trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model.](https://arxiv.org/abs/2211.05100)
  - [You can also see a high-level overview of the model here](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4)
- [Vector Space Models - Series of lessons from DeepLearning.AI's Natural Language Processing specialization discussing the basics of vector space models and their use in language modeling.](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3)

### Pre-training and Scaling Laws

- [Scaling Laws for Neural Language Models - empirical study by researchers at OpenAI exploring the scaling laws for large language models.](https://arxiv.org/abs/2001.08361)

### Model Architectures and Pre-Training Objectives

- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization? - The paper examines modeling choices in large pre-trained language models and identifies the optimal approach for zero-shot generalization.](https://arxiv.org/pdf/2204.05832.pdf)
- [Hugging Face Tasks](https://huggingface.co/tasks) and [Model Hub](https://huggingface.co/models) - Collection of resources to tackle varying machine learning tasks using the HuggingFace library.
- [LLaMA: Open and Efficient Foundation Language Models - Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)](https://arxiv.org/pdf/2302.13971.pdf)

### Scaling Laws and Compute-optimal Models

- [Language Models are Few-Shot Learners - This paper investigates the potential of few-shot learning in Large Language Models.](https://arxiv.org/pdf/2005.14165.pdf)
- [Training Compute-Optimal Large Language Models - Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.](https://arxiv.org/pdf/2203.15556.pdf)
- [BloombergGPT: A Large Language Model for Finance - LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.](https://arxiv.org/pdf/2303.17564.pdf)

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
  - _Unidirectional_ - must build up knowledge by a massive amount of examples as they have no context AFTER the token.
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
- **2D parallelism:** combining data parallelism with pipeline parallelism
- **3D parallelism:** combining data parallelism with _BOTH_ pipeline parallelism and tensor parallelism simultaneously

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

---

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

---

## Week 1 - Lab

- Dialogue summarization task using generative AI
- Input text affects output of model
- Compare zero, one, and few shot inferences
- Spins up an AWS instance -> Sagemaker Studio -> Jupyter Lab
- Copy premade notebook/files over
- Notebook verifies size of instance so we can do the work (m5_2xlarge)
- Install pip, tensorflow, keras, torch (pytorch), torchdata, and then hugging face's datasets, transformers, evaluate, rouge_score, and peft libraries
- Generating a summary of a dialogue with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face
  - [Hugging face `transformers` package](https://huggingface.co/docs/transformers/index)
  - [Hugging face `dialogsum` dataset](https://huggingface.co/datasets/knkarthick/dialogsum)

---

---

## Week 2 - Tuning

- Can train a model off a large dataset from the web
- Then _fine-tune_ the model with a specific set of data for your purpose
  - **Catastrophic forgetting:** must look out for, when the model forgets a big chunk of that language/data from before.
- 2 Types
  - **Instruction Fine Tuning:**
  - **PEP (Parameter Efficient) Fine Tuning:** Specialized Application usage allowing you to freeze specific tokens/data so it isn't lost

---

## Instruction Fine Tuning

- _fine-tuning_ with instruction prompts (i.e. **instruction fine-tuning**)
- The purpose of _fine-tuning_ is to improve the performance and adaptability of a pre-trained language model for specific tasks.
- Drawbacks of few-shot: examples in prompt take up space in context window for other helpful information
- Typically, models learn/build of a **self-supervised** process consuming a large amount of data.
- A **Supervised** learning process includes a dataset of labeled examples to update the weights of the LLM.
  - It includes **Prompt|Completion** pairs
  - Specific instructions for the model. For each pair, if:
    - Summarizing: `Summarize the following text:`
    - Translating: `Translate this sentence to...`
- Requires enough memory to store and run everything that we learned about with `pre-training`

1. **PREPARE YOUR TRAINING DATA:** developers have prompt template libraries that we can use WITH large dataset WITHOUT prompts to create fine-tuning data.
2. **TRAINING SPLITS:** divide data into _training, validation, and test_ datasets.
3. **TRAINING**
   a) Compare the LLM's completion sentiment (probability distribution) with the actual training label using a standard `cross-entropy` function to calculate loss
   b) Use calculated loss to update weights in LLM
4. **VALIDATION**
   a) Separate steps to give LLM validation accuracy
5. **TEST**
   a) Used to give test accuracy

### Single Task Fine-tuning

- Often, 500-1000 examples needed to improve the model greatly with fine-tuning
- **Catastrophic forgetting:** Fine tuning can improve the model in specific areas, but the model may forget how to do other tasks that were not in the tuning. This is a common problem in ML, especially deep learning models
  - If it doesn't impact your use case, (i.e. needing the additional task processing) then it is probably ok
  - If you need multiple task processing, you may need to do more fine tuning on _MULTIPLE TASKS_ AT THE SAME TIME (see section below)
  - One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training.
    - **Parameter Efficient Fine Tunning (PEFT)** preserves weights of the original LLM while training just a small set of adaptive layers

### Multi-task Instruction Fine-Tuning

- Include multiple instructions in a single iteration
  - i.e. Summarize the following text: <text> Rate this review: <review> Translate into Python code: <psuedo code> Identify the places: <text>
- Downside: Requires A LOT of data (50-100k examples)
- **FLAN** Family of models
  - **Fine-tuned Language Net (FLAN):** Specific set of instructions used to fine-tune models - **FLAN-T5:** has been fined-tuned on 473 datasets
    ![Flan T5 - Instruction Datasets](flan-t5.png)
- **SAMsum** - a dialog dataset used to create a high-quality summarization model by linguists. This is included in **FLAN-T5**
  - Include multiple ways to ask the same question will help the model understand
- **dialogsum** is another dataset (13,000+ chatbot supporting dialog summaries). When used with _fine-tuning_
  - Using your OWN company's conversations can help _fine-tuning_ immensely

### Model Evaluation

- How can you evaluate your _fine-tuned_ model over the _pre-trained_ model
- Accuracy = Correct Predictions / Total Predictions
- **ROUGE (SUMMARIES) ** and **BLEU SCORE (TRANSLATIONS)**
  - **Recal Oriented Under Study for Jesting Evaluation (ROUGE):** access the quality of automatically generated summaries by comparing them to human-generated reference summaries
    - unigram - one word
      - Looks at number of unigrams matching output to the reference as well as the total count of unigrams in both
      - Will miss things like the word _not_ in a sentence
    - bigram - two words
      - **ROUGE-2** - 2 word pairs in output vs reference
    - n-gram - n words
      - **ROUGE-L** - Length of the longest common subsequence (LCS) between the output and reference
      - Matches (2/4) + Total Count (2/5) / 2 = 0.44
  - **Bilingual Evaluation Understudy (BLEU):** an algorithm designed to evaluate the quality of machine-translated text by comparing it to human-generated translations
    - AVG(precision across range of n-gram sizes) - similar calcuation as **ROUGE**
      - REF: I am very happy to say that I am drinking a warm cup a tea.
      - MODEL: I am very happy that I am drinking a cup of tea. - BLEU 0.495

### Model Benchmarks

- Using pre-existing dataset and associated benchmarks already established by LLM researchers
- Select datasets that isolate specific model skills as well as focus on potential risks like disinformation or copyright infringement
- **GLUE, SuperGLUE, HELM, MMLU, BIG-bench** are all benchmarks
- **GLUE (General language understanding evaluation) (2017):** can be used to measure and compare model performance
- Have leaderboards to compare and contrast models
- **Massive Multitask Language Understanding (MMLU) (2021):** Tested on math, computer science, law, etc
- **BIG-Bench (2022):** 204 tasks linguistics, childhood development, math, reasoning, physics, social bias, software development, etc
  - Multiple levels to keep costs down
- **Holistic Evaluation of Language Models (HELM):** Various metrics are measured including accuracy, calibration, robustness, _fairness, bias, toxicity,_ and efficiency

---

## Parameter Efficient Fine-Tuning

- **Memory Usage:** 12-20x weights: Trainable weights, optimizer states, gradients, forward activations, temp memory
  - Too large to handle on consumer hardware
- **PEFT:** only update a small subset of trainable layers/components while _freezing_ others
  - Weights are frozen and other layers are only
  - As only _a small subset is updated_ you can avoid **Catastrophic forgetting**
- Tradeoffs of PEFT: _Memory Efficiency, Parameter Efficiency, Training Speed, Model Performance, and Inference Costs_

### PEFT Methods

- **SELECTIVE:** select subset of initial LLM parameters to fine-tune
- **REPARAMETERIZATION:** reparameterize model weights using a low-rank representation (**LoRA**)
- **ADDITIVE:** Add trainable layers or parameters to the model
  - Adapters are one way of adding
  - Soft prompts are another (_Prompt Tuning_)

### LoRA

1. Freeze most of the original LLM weights prior to the `self-attention` step of the **Encoder**
2. Inject 2 **rank decomposition matrices** whose product is the same size as the original LLM
3. Train the weights of the smaller matrices
   a. Matrix multiply the low rank matrices
   b. Add the result of the matrix multiplication to the original weights

- Practical Example based on the _Attention is All You Need_ paper
  - _Using the base Transformer model presented in the paper_, Transformer weights have dimensions d x k = 512 x 64
  - So 512 x 64 = **32,768 trainable parameters**
- Using **LoRA** with rank r = 8
  - A has dimensions r x k = 8 x 64 = 512 parameters
  - B has dimensions r x d = 8 x 512 = 4096 parameters
  - Then you do the matrix multiplication...etc
  - _Results in an 86% reduction in parameters to train!_
- Can use different **LoRA** _matrices to train for many different tasks_
  - Much smaller and easier to store than the entire model
- Comparing **FLAN-T5** full-tuning to **LoRA** tuning, while not AS accurate, are only a few percentage points off and still improve overall accuracy quite a bit (with a lower footprint)
- While the field is obviously still evolving, a **LoRA rank** of 16 to 512 appear to have the best accuracy when evaluated with **BLEU** and **ROUGE**

### Prompt Tuning / Soft Prompts

- **NOT** Prompt Engineering
  - None, one-shot, or few-shot inference
  - Goal is to get the model to figure out what you're trying to get it to do
  - Limited to the length of the context window
- _With **Prompt Tuning**, you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimized values_
  - Same length as token vectors
  - Tokens that represent natural language are hard - exists at a unique point in multi-dimensional space
  - Soft Prompts are virtual tokens that can take on any value in the continuous multi-dimensional embedding space
  - Underlying model is not updated (weights are frozen)
  - Instead, embedding vectors of the soft-prompt are updated over time
  - Only 10k-100k of parameters will be updated whereas with _full fine-tuning_ it will be in the millions to billions of parameters updated.
- Train a set of soft prompts for multiple tasks
  - Prepend your input prompt with the learn tokens
  - To switch to another task, swap out the prepended value with the new soft prompt
- Prompt tuning is more effective with larger models
