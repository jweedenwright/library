# Generative AI with Large Language Models (LLM)

- **PREREQS: Basic data science and Python**
- **Course Objectives**: explore steps in generative ai lifecycle: scoping, select model, optimize model for deployment, and integrate into application. Specifically around LLM (so text primarly)
- [FAQ for GenAI with LLMs Labs](https://community.deeplearning.ai/t/genai-with-llms-lab-faq/374869)
- If you cannot find an answer in the FAQs, you can [search for or create a new post here](https://community.deeplearning.ai/c/course-q-a/generative-ai-with-large-language-models/328)

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

## Contributors

- Ehsan Kamalinejad, Ph.D. - Machine Learning Applied Scientist, AWS
- Nashlie Sephus, Ph.D. - Principal Technology Evangelist for Amazon AI, AWS
- Saleh Soltan, Ph.D. - Senior Applied Scientist, Amazon Alexa
- Heiko Hotz - Senior Solutions Architect for AI & Machine Learning, AWS
- Philipp Schmid - Technical Lead, Hugging Face and AWS ML Hero

## Week 1 Intro

- **Transformer architecture**: Looking at really important parts to get the intuition you need to make practical use of these models
  - Make foundation around vision transformers

### Generative AI Project Lifecycle

1. Decide if using a foundational model _off the shelf_ or if you're _pre-training_ your own model (type of model, size of model 100 billion or bigger VS 1 to 30 billion VS sub-billion parameter model)

   - Big difference between google AI that needs to know everything VS a chat bot for a single small company

2. Do you want to _fine tune_ and _customize_ that model for your specific data

## Generative AI & LLMs

- A machine capable of making content that mimics human capability
- Machines find _statistical patterns_ in tons of datasets created by humans
- Foundational / Base LLMs
  - Larger parameters: GPT, BLOOM (176billion), _FLAN-T5_ (will use this one)
  - Smaller: LLaMa, PaLM, BERT (110million params in Bert-base)
  - **Parameters** - think of it as MEMORY and the larger the parameters the larger the memory/ability to figure things out

* Syntax/Writing code (_typical programming_) vs ...

### INFERENCE - Prompts and Completions

- **Inference** is the act of using a model to generate text
- You **PROMPT** a model with a question
- Model output (**COMPLETION**) _predicts the next words_ and creates output including the original prompt and the answer.

## Use Cases

- Chatbot, but this is overused as an example
- Write an essay
- Translate languages
- Translate natural language to programming code
- Identify people/places in an article
- Connect to external APIs and then process that information to respond (i.e. check a flight)

## Text Generation Before Transformers

- RNN (recurrent neural networks) were used for text generation in the past
  - Used the previous few words to predict the text and would often get it wrong
- Many words are hominyms that have multiple meanings so **context** is so very important
- **2017 - _Attention Is All You Need_ (Univ Toronto and Google)** - Transformers arrived
  - Scale efficiently
  - Parallel process
  - Learn / pay attention to words/input

## Transformers Architecture

![General Architecture](architecture.png)

- Ability to learn the relevance and context of **all** of the words in a sentence. Not just to it's neighbor, but to other words in the sentence.
- This creates an _attention map_ is created between words. This is called `self-attention`
- 2 distinct parts: **encoder** and **decoder**
- Ultimately, these models are just huge calculators and, as such, you must first **tokenize** the words and provide values for each.

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

## Generating Text with Transformers

- Walk through a translation task or a _sequence-to-sequence_ task (which was the original bojective of the transformer architecture designers)
- Translate French to English
- Data that leaves the encoder goes to the _middle_ of the decoder to _influence_ it's self-attention.
- A _start of sequence token_ is added to the input to the decoder, triggering the decoder to predict the next token, based on it's contextual understanding provided by the encoder.
- After the `softmax` output layer, we have our **first token**
- This then loops back to the **decoder** passing in the output token, _triggering the generation of the next token_ until the model predicts an _end-of-sequence_ token.
- Tokens are then _detokenized_ into words, and you have the output
- Output from `softmax` layer can be used in many ways to influence how creative your generated text is.

### Encoder / Decoder Variations

- The **encoder** and **decoder** can be separated out into different spaces
- _sequence-to-sequence_ tasks are often done as **ENCODER-ONLY** models (BERT, less common), such as sentiment analysis
- BART and T5 have _both_ **encoder** and **decoder**
- **DECODER-ONLY** models are most commonly used today - GPT family, BLOOM, Jurassic, LLaMA and many more. Can be applied to most tasks.

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

## Generative AI Project Lifecycle

![Generative AI Project Lifecycle](generative-ai-lifecycle.png)

Be _specific_ to save time and compute costs.

1. **SCOPE** - Define use case as _narrowly_ as you can
2. **SELECT** - Choose existing model or train a model from scratch
3. **ADAPT AND ALIGN MODEL**
   - Prompt Engineering
   - Fine-tuning
   - Align with Human Feedback (additional fine tuning) - ensure your model behaves well
   - Evaluate model
4. **APPLICATION INTEGRATION**
   - Optimize and deploy model for inference
   - Augment model and build _LLM-powered applications_
