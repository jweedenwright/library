# [What is generative AI?](https://www.ibm.com/think/topics/generative-ai)

Generative AI, sometimes called gen AI, is artificial intelligence (AI) that can create original content such as text, images, video, audio or software code in response to a user’s prompt or request.

Generative AI relies on sophisticated machine learning models called deep learning models algorithms that simulate the learning and decision-making processes of the human brain. These models work by identifying and encoding the patterns and relationships in huge amounts of data, and then using that information to understand users' natural language requests or questions and respond with relevant new content.

AI has been a hot technology topic for the past decade, but generative AI, and specifically the arrival of ChatGPT in 2022, has thrust AI into worldwide headlines and launched an unprecedented surge of AI innovation and adoption. Generative AI offers enormous productivity benefits for individuals and organizations, and while it also presents very real challenges and risks, businesses are forging ahead, exploring how the technology can improve their internal workflows and enrich their products and services. According to research by the management consulting firm McKinsey, one third of organizations are already using generative AI regularly in at least one business function.¹ Industry analyst Gartner projects more than 80% of organizations will have deployed generative AI applications or used generative AI application programming interfaces (APIs) by 2026.2

## How generative AI works
For the most part, generative AI operates in three phases: 

* Training, to create a foundation model that can serve as the basis of multiple gen AI applications.
* Tuning, to tailor the foundation model to a specific gen AI application.
* Generation, evaluation and retuning, to assess the gen AI application's output and continually improve its quality and accuracy.

### Training
Generative AI begins with a foundation model, a deep learning model that serves as the basis for multiple different types of generative AI applications. The most common foundation models today are large language models (LLMs), created for text generation applications, but there are also foundation models for image generation, video generation, and sound and music generation as well as multimodal foundation models that can support several kinds content generation.

To create a foundation model, practitioners train a deep learning algorithm on huge volumes of raw, unstructured, unlabeled data e.g., terabytes of data culled from the internet or some other huge data source. During training, the algorithm performs and evaluates millions of ‘fill in the blank’ exercises, trying to predict the next element in a sequence e.g., the next word in a sentence, the next element in an image, the next command in a line of code and continually adjusting itself to minimize the difference between its predictions and the actual data (or ‘correct’ result).

The result of this training is a neural network of parameters, encoded representations of the entities, patterns and relationships in the data, that can generate content autonomously in response to inputs, or prompts.

This training process is compute-intensive, time-consuming and expensive: it requires thousands of clustered graphics processing units (GPUs) and weeks of processing, all of which costs millions of dollars. Open-source foundation model projects, such as Meta's Llama-2, enable gen AI developers to avoid this step and its costs.

### Tuning
Metaphorically speaking, a foundation model is a generalist: It knows a lot about a lot of types of content, but often can’t generate specific types of output with desired accuracy or fidelity. For that, the model must be tuned to a specific content generation task. This can be done in a variety of ways.

### Fine tuning
Fine tuning involves feeding the model labeled data specific to the content generation application questions or prompts the application is likely to receive, and corresponding correct answers in the desired format. For example, if a development team is trying to create a customer service chatbot, it would create hundreds or thousands of documents containing labeled customers service questions and correct answers, and then feed those documents to the model.

Fine-tuning is labor-intensive. Developers often outsource the task to companies with large data-labeling workforces.

### Reinforcement learning with human feedback (RLHF)
In RLHF, human users respond to generated content with evaluations the model can use to update the model for greater accuracy or relevance. Often, RLHF involves people ‘scoring’ different outputs in response to the same prompt. But it can be as simple as having people type or talk back to a chatbot or virtual assistant, correcting its output.

### Generation, evaluation, more tuning
Developers and users continually assess the outputs of their generative AI apps, and further tune the model even as often as once a week for greater accuracy or relevance. (In contrast, the foundation model itself is updated much less frequently, perhaps every year or 18 months.)

Another option for improving a gen AI app's performance is retrieval augmented generation (RAG). RAG is a framework for extending the foundation model to use relevant sources outside of the training data, to supplement and refine the parameters or representations in the original model. RAG can ensure that a generative AI app always has access to the most current information. As a bonus, the additional sources accessed via RAG are transparent to users in a way that the knowledge in the original foundation model is not.

## Generative AI model architectures and how they have evolved
Truly generative AI models deep learning models that can autonomously create content on demand have evolved over the last dozen years or so. The milestone model architectures during that period include

* Variational autoencoders (VAEs), which drove breakthroughs in image recognition, natural language processing and anomaly detection.
* Generative adversarial networks (GANs) and diffusion models, which improved the accuracy of previous applications and enabled some of the first AI solutions for photo-realistic image generation.
* Transformers, the deep learning model architecture behind the foremost foundation models and generative AI solutions today.

### Variational autoencoders (VAEs)
An autoencoder is a deep learning model comprising two connected neural networks: One that encodes (or compresses) a huge amount of unstructured, unlabeled training data into parameters, and another that decodes those parameters to reconstruct the content. Technically, autoencoders can generate new content, but they’re more useful for compressing data for storage or transfer, and decompressing it for use, than they are for high-quality content generation.

Introduced in 2013, variational autoencoders (VAEs) can encode data like an autoencoder, but decode multiple new variations of the content. By training a VAE to generate variations toward a particular goal, it can ‘zero in’ on more accurate, higher-fidelity content over time. Early VAE applications included anomaly detection (e.g., medical image analysis) and natural language generation.

### Generative adversarial networks (GANs)
GANs, introduced in 2014, also comprise two neural networks: A generator, which generates new content, and a discriminator, which evaluates the accuracy and quality the generated data. These adversarial algorithms encourages the model to generate increasingly high-quality outpits.

GANs are commonly used for image and video generation, but can generate high-quality, realistic content across various domains. They've proven particularly successful at tasks as style transfer (altering the style of an image from, say, a photo to a pencil sketch) and data augmentation (creating new, synthetic data to increase the size and diversity of a training data set).

### Diffusion models
Also introduced in 2014, diffusion models work by first adding noise to the training data until it’s random and unrecognizable, and then training the algorithm to iteratively diffuse the noise to reveal a desired output.

Diffusion models take more time to train than VAEs or GANs, but ultimately offer finer-grained control over output, particularly for high-quality image generation tool. DALL-E, Open AI’s image-generation tool, is driven by a diffusion model.

### Transformers
First documented in a 2017 paper published by Ashish Vaswani and others, transformers evolve the encoder-decoder paradigm to enable a big step forward in the way foundation models are trained, and in the quality and range of content they can produce. These models are at the core of most of today’s headline-making generative AI tools, including ChatGPT and GPT-4, Copilot, BERT, Bard, and Midjourney to name a few.

Transformers use a concept called attention, determining and focusing on what’s most important about data within a sequence to;
* process entire sequences of data e.g., sentences instead of individual words simultaneously;
* capture the context of the data within the sequence;
* encode the training data into embeddings (also called hyperparameters) that represent the data and its context.

In addition to enabling faster training, transformers excel at natural language processing (NLP) and natural language understanding (NLU), and can generate longer sequences of data e.g., not just answers to questions, but poems, articles or papers with greater accuracy and higher quality than other deep generative AI models. Transformer models can also be trained or tuned to use tools e.g., a spreadsheet application, HTML, a drawing program to output content in a particular format.

## What generative AI can create
Generative AI can create many types of content across many different domains. 

### Text
Generative models. especially those based on transformers, can generate coherent, contextually relevant text, everything from instructions and documentation to brochures, emails, web site copy, blogs, articles, reports, papers, and even creative writing. They can also perform repetitive or tedious writing tasks (e.g., such as drafting summaries of documents or meta descriptions of web pages), freeing writers’ time for more creative, higher-value work.

### Images and video
Image generation such as DALL-E, Midjourney and Stable Diffusion can create realistic images or original art, and can perform style transfer, image-to-image translation and other image editing or image enhancement tasks. Emerging gen AI video tools can create animations from text prompts, and can apply special effects to existing video more quickly and cost-effectively than other methods.

### Sound, speech and music
Generative models can synthesize natural-sounding speech and audio content for voice-enabled AI chatbots and digital assistants, audiobook narration and other applications. The same technology can generate original music that mimics the structure and sound of professional compositions.

### Software code
Gen AI can generate original code, autocomplete code snippets, translate between programming languages and summarize code functionality. It enables developers to quickly prototype, refactor, and debug applications while offering a natural language interface for coding tasks.

### Design and art
Generative AI models can generate unique works of art and design, or assist in graphic design. Applications include dynamic generation of environments, characters or avatars, and special effects for virtual simulations and video games.

### Simulations and synthetic data
Generative AI models can be trained to generate synthetic data, or synthetic structures based on real or synthetic data. For example, generative AI is applied in drug discovery to generate molecular structures with desired properties, aiding in the design of new pharmaceutical compounds.


# [Natural Language Processing - NLP](https://www.ibm.com/think/topics/natural-language-processing)

Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language.

NLP enables computers and digital devices to recognize, understand and generate text and speech by combining computational linguistics, the rule-based modeling of human language together with statistical modeling, machine learning and deep learning.

NLP research has helped enable the era of generative AI, from the communication skills of large language models (LLMs) to the ability of image generation models to understand requests. NLP is already part of everyday life for many, powering search engines, prompting chatbots for customer service with spoken commands, voice-operated GPS systems and question-answering digital assistants on smartphones such as Amazon’s Alexa, Apple’s Siri and Microsoft’s Cortana.

NLP also plays a growing role in enterprise solutions that help streamline and automate business operations, increase employee productivity and simplify business processes.

## Benefits of NLP

NLP makes it easier for humans to communicate and collaborate with machines, by allowing them to do so in the natural human language they use every day. This offers benefits across many industries and applications.

* Automation of repetitive tasks
* Improved data analysis and insights
* Enhanced search
* Content generation

### Automation of repetitive tasks 

NLP is especially useful in fully or partially automating tasks like customer support, data entry and document handling. For example, NLP-powered chatbots can handle routine customer queries, freeing up human agents for more complex issues. In document processing, NLP tools can automatically classify, extract key information and summarize content, reducing the time and errors associated with manual data handling. NLP facilitates language translation, converting text from one language to another while preserving meaning, context and nuances.
Improved data analysis

NLP enhances data analysis by enabling the extraction of insights from unstructured text data, such as customer reviews, social media posts and news articles. By using text mining techniques, NLP can identify patterns, trends and sentiments that are not immediately obvious in large datasets. Sentiment analysis enables the extraction of subjective qualities, attitudes, emotions, sarcasm, confusion or suspicion from text. This is often used for routing communications to the system or the person most likely to make the next response.

This allows businesses to better understand customer preferences, market conditions and public opinion. NLP tools can also perform categorization and summarization of vast amounts of text, making it easier for analysts to identify key information and make data-driven decisions more efficiently.

###  Enhanced search

NLP benefits search by enabling systems to understand the intent behind user queries, providing more accurate and contextually relevant results. Instead of relying solely on keyword matching, NLP-powered search engines analyze the meaning of words and phrases, making it easier to find information even when queries are vague or complex. This improves user experience, whether in web searches, document retrieval or enterprise data systems.

### Powerful content generation

NLP powers advanced language models to create human-like text for various purposes. Pre-trained models, such as GPT-4, can generate articles, reports, marketing copy, product descriptions and even creative writing based on prompts provided by users. NLP-powered tools can also assist in automating tasks like drafting emails, writing social media posts or legal documentation. By understanding context, tone and style, NLP sees to it that the generated content is coherent, relevant and aligned with the intended message, saving time and effort in content creation while maintaining quality.

## Approaches to NLP

NLP combines the power of computational linguistics together with machine learning algorithms and deep learning. Computational linguistics uses data science to analyze language and speech. It includes two main types of analysis: syntactical analysis and semantical analysis. Syntactical analysis determines the meaning of a word, phrase or sentence by parsing the syntax of the words and applying preprogrammed rules of grammar. Semantical analysis uses the syntactic output to draw meaning from the words and interpret their meaning within the sentence structure.

The parsing of words can take one of two forms. Dependency parsing looks at the relationships between words, such as identifying nouns and verbs, while constituency parsing then builds a parse tree (or syntax tree): a rooted and ordered representation of the syntactic structure of the sentence or string of words. The resulting parse trees underly the functions of language translators and speech recognition. Ideally, this analysis makes the output either text or speech understandable to both NLP models and people.

Self-supervised learning (SSL) in particular is useful for supporting NLP because NLP requires large amounts of labeled data to train AI models. Because these labeled datasets require time-consuming annotation, a process involving manual labeling by humans, gathering sufficient data can be prohibitively difficult. Self-supervised approaches can be more time-effective and cost-effective, as they replace some or all manually labeled training data.
 
Three different approaches to NLP include:

### Rules-based NLP

The earliest NLP applications were simple if-then decision trees, requiring preprogrammed rules. They are only able to provide answers in response to specific prompts, such as the original version of Moviefone, which had rudimentary natural language generation (NLG) capabilities. Because there is no machine learning or AI capability in rules-based NLP, this function is highly limited and not scalable.

### Statistical NLP

Developed later, statistical NLP automatically extracts, classifies and labels elements of text and voice data and then assigns a statistical likelihood to each possible meaning of those elements. This relies on machine learning, enabling a sophisticated breakdown of linguistics such as part-of-speech tagging.
 
Statistical NLP introduced the essential technique of mapping language elements, such as words and grammatical rules to a vector representation so that language can be modeled by using mathematical (statistical) methods, including regression or Markov models. This informed early NLP developments such as spellcheckers and T9 texting (Text on 9 keys, to be used on Touch-Tone telephones).

### Deep learning NLP

Recently, deep learning models have become the dominant mode of NLP, by using huge volumes of raw, unstructured data both text and voice to become ever more accurate. Deep learning can be viewed as a further evolution of statistical NLP, with the difference that it uses neural network models. There are several subcategories of models:

- Sequence-to-Sequence (seq2seq) models: Based on recurrent neural networks (RNN), they have mostly been used for machine translation by converting a phrase from one domain (such as the German language) into the phrase of another domain (such as English).

- Transformer models: They use tokenization of language (the position of each token words or subwords) and self-attention (capturing dependencies and relationships) to calculate the relation of different language parts to one another. Transformer models can be efficiently trained by using self-supervised learning on massive text databases. A landmark in transformer models was Google’s bidirectional encoder representations from transformers (BERT), which became and remains the basis of how Google’s search engine works.

- Autoregressive models: This type of transformer model is trained specifically to predict the next word in a sequence, which represents a huge leap forward in the ability to generate text. Examples of autoregressive LLMs include GPT, Llama, Claude and the open-source Mistral.

- Foundation models: Prebuilt and curated foundation models can speed the launching of an NLP effort and boost trust in its operation. For example, the IBM® Granite™ foundation models are widely applicable across industries. They support NLP tasks including content generation and insight extraction. Additionally, they facilitate retrieval-augmented generation, a framework for improving the quality of response by linking the model to external sources of knowledge. The models also perform named entity recognition which involves identifying and extracting key information in a text.

## NLP Tasks

Several NLP tasks typically help process human text and voice data in ways that help the computer make sense of what it’s ingesting. Some of these tasks include:

* Coreference resolution
* Named entity recognition
* Part-of-speech tagging
* Word sense disambiguation

### Coreference resolution

This is the task of identifying if and when two words refer to the same entity. The most common example is determining the person or object to which a certain pronoun refers (such as “she” = “Mary”). But it can also identify a metaphor or an idiom in the text (such as an instance in which “bear” isn’t an animal, but a large and hairy person). 

### Named entity recognition (NER)

NER identifies words or phrases as useful entities. NER identifies “London” as a location or “Maria” as a person's name.

### Part-of-speech tagging

Also called grammatical tagging, this is the process of determining which part of speech a word or piece of text is, based on its use and context. For example, part-of-speech identifies “make” as a verb in “I can make a paper plane,” and as a noun in “What make of car do you own?”

### Word sense disambiguation

This is the selection of a word meaning for a word with multiple possible meanings. This uses a process of semantic analysis to examine the word in context. For example, word sense disambiguation helps distinguish the meaning of the verb “make” in “make the grade” (to achieve) versus “make a bet” (to place). Sorting out “I will be merry when I marry Mary” requires a sophisticated NLP system.

## How NLP works

NLP works by combining various computational techniques to analyze, understand and generate human language in a way that machines can process. Here is an overview of a typical NLP pipeline and its steps:

### Text preprocessing

NLP text preprocessing prepares raw text for analysis by transforming it into a format that machines can more easily understand. It begins with tokenization, which involves splitting the text into smaller units like words, sentences or phrases. This helps break down complex text into manageable parts. Next, lowercasing is applied to standardize the text by converting all characters to lowercase, ensuring that words like "Apple" and "apple" are treated the same. Stop word removal is another common step, where frequently used words like "is" or "the" are filtered out because they don't add significant meaning to the text. Stemming or lemmatization reduces words to their root form (e.g., "running" becomes "run"), making it easier to analyze language by grouping different forms of the same word. Additionally, text cleaning removes unwanted elements such as punctuation, special characters and numbers that may clutter the analysis.

After preprocessing, the text is clean, standardized and ready for machine learning models to interpret effectively.

### Feature extraction

Feature extraction is the process of converting raw text into numerical representations that machines can analyze and interpret. This involves transforming text into structured data by using NLP techniques like Bag of Words and TF-IDF, which quantify the presence and importance of words in a document. More advanced methods include word embeddings like Word2Vec or GloVe, which represent words as dense vectors in a continuous space, capturing semantic relationships between words. Contextual embeddings further enhance this by considering the context in which words appear, allowing for richer, more nuanced representations.

### Text analysis

Text analysis involves interpreting and extracting meaningful information from text data through various computational techniques. This process includes tasks such as part-of-speech (POS) tagging, which identifies grammatical roles of words and named entity recognition (NER), which detects specific entities like names, locations and dates. Dependency parsing analyzes grammatical relationships between words to understand sentence structure, while sentiment analysis determines the emotional tone of the text, assessing whether it is positive, negative or neutral. Topic modeling identifies underlying themes or topics within a text or across a corpus of documents. Natural language understanding (NLU) is a subset of NLP that focuses on analyzing the meaning behind sentences. NLU enables software to find similar meanings in different sentences or to process words that have different meanings. Through these techniques, NLP text analysis transforms unstructured text into insights.

### Model training

Processed data is then used to train machine learning models, which learn patterns and relationships within the data. During training, the model adjusts its parameters to minimize errors and improve its performance. Once trained, the model can be used to make predictions or generate outputs on new, unseen data. The effectiveness of NLP modeling is continually refined through evaluation, validation and fine-tuning to enhance accuracy and relevance in real-world applications.

Different software environments are useful throughout the said processes. For example, the Natural Language Toolkit (NLTK) is a suite of libraries and programs for English that is written in the Python programming language. It supports text classification, tokenization, stemming, tagging, parsing and semantic reasoning functionalities. TensorFlow is a free and open-source software library for machine learning and AI that can be used to train models for NLP applications. Tutorials and certifications abound for those interested in familiarizing themselves with such tools.

## Challenges of NLP 

Even state-of-the-art NLP models are not perfect, just as human speech is prone to error. As with any AI technology, NLP comes with potential pitfalls. Human language is filled with ambiguities that make it difficult for programmers to write software that accurately determines the intended meaning of text or voice data. Human language might take years for humans to learn and many never stop learning. But then programmers must teach natural language-powered applications to recognize and understand irregularities so their applications can be accurate and useful. Associated risks might include:

### Biased training

As with any AI function, biased data used in training will skew the answers. The more diverse the users of an NLP function, the more significant this risk becomes, such as in government services, healthcare and HR interactions. Training datasets scraped from the web, for example, are prone to bias.
Misinterpretation

As in programming, there is a risk of garbage in, garbage out (GIGO). Speech recognition, also known as speech-to-text, is the task of reliably converting voice data into text data. But NLP solutions can become confused if spoken input is in an obscure dialect, mumbled, too full of slang, homonyms, incorrect grammar, idioms, fragments, mispronunciations, contractions or recorded with too much background noise.

### New vocabulary

New words are continually being invented or imported. The conventions of grammar can evolve or be intentionally broken. In these cases, NLP can either make a best guess or admit it’s unsure and either way, this creates a complication.

### Tone of voice

When people speak, their verbal delivery or even body language can give an entirely different meaning than the words alone. Exaggeration for effect, stressing words for importance or sarcasm can be confused by NLP, making the semantic analysis more difficult and less reliable.


# [What is AI Inference?](https://cloud.google.com/discover/what-is-ai-inference)

AI inference is the "doing" part of artificial intelligence. It's the moment a trained model stops learning and starts working, turning its knowledge into real-world results.

Think of it this way: if training is like teaching an AI a new skill, inference is that AI actually using the skill to do a job. It takes in new data (like a photo or a piece of text) and produces an instant output, like a prediction, generates a photo, or makes a decision. This is where AI delivers business value. For anyone building with AI, understanding how to make inference fast, scalable, and cost-effective is the key to creating successful solutions.

## 'AI training' versus 'fine-tuning' versus 'inference' versus 'serving'

While the complete AI life cycle involves everything from data collection to long-term monitoring, a model's central journey from creation to execution has three key stages. The first two are about learning, while the last one is about putting that learning to work.

* AI training is the foundational learning phase. It's a computationally intensive process where a model analyzes a massive dataset to learn patterns and relationships. The goal is to create an accurate and knowledgeable model. This requires powerful hardware accelerators (like GPUs and TPUs) and can take anywhere from hours to weeks.
* AI fine-tuning is a shortcut to training. It takes a powerful, pre-trained model and adapts it to a more specific task using a smaller, specialized dataset. This saves significant time and resources compared to training a model from scratch.
* AI inference is the execution phase. It uses the trained and fine-tuned model to make fast predictions on new, "unseen" data. Each individual prediction is far less computationally demanding than training, but delivering millions of predictions in real-time requires a highly optimized and scalable infrastructure.
* AI serving is the process of deploying and managing the model for inference. This often involves packaging the model, setting up an API endpoint, and managing the infrastructure to handle requests.

This table summarizes the key differences:

||AI training|AI fine-tuning|AI inference|AI serving|
| --- | --- | --- | --- | --- |
| Objective | Build a new model from scratch. | Adapt a pre-trained model for a specific task. | Use a trained model to make predictions. | Deploy and manage the model to handle inference requests. |
| Process | Iteratively learns from a large dataset. | Refines an existing model with a smaller dataset. | A single, fast "forward pass" of new data. | Package the model and expose it as an API |
| Data | Large, historical, labeled datasets. | Smaller, task-specific datasets. | Live, real-world, unlabeled data. | N/A |
| Business focus | Model accuracy and capability. | Efficiency and customization. | Speed (latency), scale, and cost-efficiency. | Reliability, scalability, and manageability of the inference endpoint. |

## How does AI inference work?

At its core, AI inference involves three steps that turn new data into a useful output. 

Let's walk through it with a simple example: an AI model built to identify objects in photos.

1. Input data preparation: First, new data is provided — for instance, a photo you've just submitted. This photo is instantly prepped for the model, which might mean simply resizing it to the exact dimensions it was trained on.
2. Model execution: Next, the AI model analyzes the prepared photo. It looks for patterns — like colors, shapes, and textures — that match what it learned during its training. This quick analysis is called a "forward pass," a read-only step where the model applies its knowledge without learning anything new.
3. Output generation: The model produces an actionable result. For the photo analysis, this might be a probability score (such as a 95% chance the image contains a "dog"). This output is then sent to the application and displayed to the user.

While a single inference is quick, serving millions of users in real time adds to the latency, cost, and requires optimized hardware. AI specialized Graphics Processing Units (GPUs) and Google's Tensor Processing Units are designed to handle these tasks efficiently along with orchestration with Google Kubernetes Engine, helping to increase throughput and lower latency.

## Types of AI inference

### Cloud inference: For power and scale

This is the most common approach, where inference runs on powerful remote servers in a data center. The cloud offers immense scalability and computational resources, making it ideal for handling massive datasets and complex models. Within the cloud, there are typically two primary modes of inference:

* Real-time (online) inference: Processes individual requests instantly as they arrive, often within milliseconds. This is crucial for interactive applications that demand immediate feedback.
* Batch (offline) inference: Handles large volumes of data all at once, typically when immediate responses aren't required. It's a highly cost-effective method for periodic analyses or scheduled tasks.

### Edge inference: For speed and privacy

This approach performs inference directly on the device where data is generated — this could be on a smartphone, or an industrial sensor. By avoiding a round-trip to the cloud, edge inference offers unique advantages:

* Reduced latency: Responses are nearly instantaneous, critical for applications like autonomous vehicles or real-time manufacturing checks.
* Enhanced privacy: Sensitive data (such as medical scans, personal photos, video feeds) can be processed on-device without ever being sent to the cloud.
* Lower bandwidth costs: Processing data locally significantly reduces the amount of data that needs to be uploaded and downloaded.
* Offline functionality: The application can continue to work even without an internet connection, ensuring continuous operation in remote or disconnected environments.

## AI inference comparison

To help you choose the best approach for your specific needs, here’s a quick comparison of the key characteristics and use cases for each type of AI inference:

| Feature | Batch inference | Real-time inference | Edge inference |
| --- | --- | --- | --- |
| Primary location | Cloud (data centers) | Cloud (data centers) | Local device (such as phone, IoT sensor, robot) |
| Latency/responsiveness | High (predictions returned after processing batch) | Very low (milliseconds to seconds per request) | Extremely low (near-instantaneous, no network hop) |
| Data volume | Large datasets (such as terabytes) | Individual events/requests | Individual events/requests (on-device) |
| Data flow | Data sent to cloud, processed, results returned | Each request sent to cloud, processed, returned | Data processed on device, results used on device |
| Typical use cases | Large-scale document categorization, overnight financial analysis, periodic predictive maintenance | Product recommendations, chatbots, live translation, real-time fraud alerts | Autonomous driving, smart cameras, offline voice assistants, industrial quality control |
| Key benefits | Cost-effective for large, non-urgent tasks | Immediate responsiveness for user-facing apps | Minimal latency, enhanced privacy, offline capability, reduced bandwidth costs |

# Other Links
* [AI Inference: The Secret to AI’s Superpower (11min video)](https://www.youtube.com/watch?v=XtT5i0ZeHHE&t=19s)
* [Open vs Closed Models in AI (8min video)](https://www.youtube.com/watch?v=vRtHC5YgN1w)
* [Prompt Engineering: Temperature, Top-K, Top-P (8min video)](https://www.youtube.com/watch?v=-BBulGM6xF0)
* [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
