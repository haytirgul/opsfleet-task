# RAG Course

## Module 1 - RAG Overview

**Retriever** - the engine that searches the data from a database to augment the information the LLM sees when answered.

## Module 2 - Information retrieval and search foundations

### Search techniques:
* **Hybrid search** - semantic and keyword based, the output document is combined and ranked.
* **Keyword based search**
* **Semantic search (embeddings)** - Models map text to vectors. Popular choices: **OpenAI text-embedding-3**, **Cohere embed-english-v3.0**, **BAAI/bge-m3** (Open Source).
* **Metadata filtering** - filtering documents from the research before hand, for example, if HR is searching for document the system will block any engineering documents (both PII and retrieval contamination reasons)

### Metadata filtering
* Metadata is information on the data, for example author, publication date, title, etc.
* Metadata filtering is not a search technique, and can not handle retrieval alone, but provides a lot of value when combined with active search techniques.

### Keyword search (TF-IDF)
* When a document contains more words from the prompt, it’s relevancy increase, for example, if a prompt asks: “how to make a pizza”, and a document contains 10 times the word pizza, than it’s relevancy increases above documents that don’t contain the word or contains it fewer times.
* In this scenario, the prompt and document is treated as a bag of words, where word order is not important, only the number of word instances.
* The words count are held in a vector, where each spot in the vector represent a word in the vocabulary (we don’t talk currently on software optimization such as holding it as dict, or linguistic optimizations such as using stemming children -> kid, pizzas -> pizza etc)
* The docs are organised in a matrix where rows are the words and columns are the docs, where index [i,j] is the number of occurrences of words in a document. This matrix is called inverted index, and allows to extract all documents holding the words simultaneously.
* In this method each document that contains at least one time the word is receiving one point for each word (regardless of how many times the word is mentioned in the doc) and the docs with the highest number of points are retrieved.
* **Frequency based scoring (TF)** - a scoring method where each instance of the word in the document adds one point to the total score.
* **Normalizing TF scores** - the TF scoring introduces a new problem, where longer documents receive more points due to their length. The normalization can be done by length of the doc.
* **TF-IDF (inverse document frequency):** in the regular TF method each word is treated equally, for example the words [the, a, an, with, …] are scored the same as [pizza, oven, pool,...] that contain much more relation context.
* In order to fix this bais in the TF, the TF-IDF we provide weight to each word found. The normalization is working that way: for each word in the vocabulary we search for how many documents it appears in, and normalise it with total docs. The rarest words will receive lower DF (document frequency) score, while more common words will receive higher value. The IDF is 1/DF resulting in higher value to rare words and lower for common. The issue that might rise is that rare words provide high points, then the solution is applying log to the IDF.

### Keywords search - BM25 (best matching 25)
A scoring method calculated as:
$$IDF \cdot \frac{TF \cdot (k1+1)}{TF + k1 \cdot (1-b + b \cdot \frac{doc\_len}{avg(all\_docs\_len)})}$$

* The score is for single word score
* The sum scores of all words in the doc is the final result for the doc

#### How does BM25 improve TF-IDF?
1. **Term frequency saturation:**
   * In the TF-IDF the number of instances of the same words increase the document relevance score in a linear manner, which means that according to the TF-IDF a document that has 20 times the word “pizza” is two-times relevant than a document that has only ten.
   * The BM25 applies thresholding (saturation), and penalises the linear increment of relevancy for the document. This means after a certain number of occurrences, adding more "pizza" doesn't increase the score much.
2. **Document length normalization:**
   * The TF-IDF penalises long documents too aggressively while the BM25 applies a smaller penalty to longer documents.
   * The result of the BM25 is that long documents are still scoring highly if they have high frequency of the keywords.

#### BM25 tunable parameters:
* **K1 - term frequency saturation**
    * Controls how much the frequency influence the score
    * Usually between 1.2-2
    * Higher values increase impact of term frequency, lower values reduce it
* **B - length normalization**
    * Controls the degree of normalization for document length
    * Range: 0 (no normalization) to 1 (full normalization)
    * Effect: balance favoring shorts vs longer documents

The BM25's final score can be viewed as estimating the total information content that document d provides about query q. This is essentially asking: "How much total information (in bits) does this document contain that's relevant to understanding this query?"

### Semantic search:
Using embedding model

#### Dot product:
* Measure direction alignment (i.e. the degree of direction of vectors) and magnitude of vectors (their size)

**When to use?**
* Vector magnitudes contain meaningful information (e.g., confidence, importance)
* You want to give higher weight to longer vectors
* Working with already normalized embeddings (same magnitude)
* Computational efficiency is critical (slightly faster)
* In attention mechanisms where absolute values matter

#### Cosine similarity:
* Measure only the direction alignment of vectors (the angle between them)
* The vectors size are normalised

**When to use?**
* Only directional similarity matters, not magnitude
* Comparing embeddings of different scales or from different sources
* Comparing texts/documents of different lengths
* You need a bounded similarity measure (-1 to 1)
* Embedding magnitudes might be affected by factors unrelated to semantic meaning
* Setting consistent similarity thresholds across different types of data

In practice, many modern embedding systems normalize vectors to unit length during preprocessing, which makes dot product and cosine similarity equivalent (differing only by a constant factor).

### Hybrid search
Hybrid search combines all the three search methods: metadata data, similarity and semantic.

**Flow:**
1. Metadata filtering filter relevant documents
2. Keyword search and semantic search each provide list of documents sorted by similarity (each method score the documents itself regardless of other method)
3. An fusion algorithm to combine the list is used

#### Reciprocal rank fusion (RRF):
* Reward document for being highly ranked on each list
* Control weight of keyword vs semantic search ranking
* Score a document by it’s rank from the list its coming from by  the following method: Assume we have N lists (N retrievers) the rank of the document will be: 
$$ \sum \frac{1}{K + \text{rank in list i}} $$
* For now assuming that K = 0. If a doc appears 2nd in one list, and 10th in the second one, we will get total score of 0.5 + 0.1 = 0.6
* Using this method we can rank all documents in our lists

**Why do we need K?** The short answer is for regularization, we don’t want a document that in the first place to be 10 times more relevant than the document in 10th place. Thus a k = 50 is good starter.

**The beta param:** defines the weight of the method, for example, we can set beta to 0.8 which means that the semantic score gets 80% of decision power vs 20% of keywords.

**Top_k** - the k highest ranked documents after ranking

### How to evaluate retrieval:
Such as other evaluation systems in learning, we need to have a ground base, which means we need to build a dataset of prompts, naked results and ground truth.
We evaluate the retrieval using precision and recall.

* **precision** - measure how many of the returned documents are relevant.
* **Recall** - measure how many  of the relevant documents returned.

#### What each metric means in RAG ranking:
* **Precision** - penalises for returning irrelevant information, for example if the retrieval returned 9 relevant docs out of 15, its means that 6 documents are not relevant, (9/15)
* **Recall** - penalised for NOT retrieving relevant information. For example if 10 relevant documents were needed, and the engine returns only 8, the recall is 80%.

We want the recall and precision to be as high as possible.
To get 100% recall and precision we need to rank highly only the relevant docs, and return only them.
Since the recall and precision are influenced by the number of retrieved documents (i.e. one retrieval returns 15, while other returns 10) it means that results are biased. To resolve this issue, we use the top k to evaluate retrieval.

**MAP @ k** - mean average precision for relevant documents in the first k documents.

**NDCG (Normalized Discounted Cumulative Gain):**
* A measure of ranking quality that handles **graded relevance** (e.g., relevant, somewhat relevant, irrelevant) rather than just binary.
* It assumes highly relevant documents are more useful when appearing earlier in the result list.
* Commonly used when you have ground truth with relevance scores (e.g. 1-5).

**How it’s evaluated:** the top k documents are given each a score, which is accumulated relevant documents based on the location, for example if the first, third and fifth are relevant documents:
* The first receive 1/1 points
* The third receive ⅔ points (as its second relevant document but ordered third)
* The fifth document receives ⅗ as it third document but ordered fifth
* The total score is: (1 + 0.66 + 0.6)/3 = 0.75

**More generic:**
* Sum precision for relevant docs for each doc
* Divide by number of docs

In order to evaluate the average precision of the engine for a given prompt, we average the MAP @ k for N prompts.

**Reciprocal rank** - measure the rank of the first relevant document in the returned list
* Reciprocal rank = 1 / rank
* This means that if the first relevant document is at rank 1 in top k (i.e appear first) the reciprocal rank would be 1, if it appears in second place it would be 0.5

We use the RR (Reciprocal rank) across many prompt to get MRR (Mean RR).
The MRR measure how soon you will find the relevant document.

#### How to use retriever metrics:
* **recall / recall @ k** - most cited metric, captures fundamental goal of finding relevant documents.
* **Precision and MAP** - asses irrelevant documents and ranking effectiveness
* **MRR** - How well the model performs at the very top of ranking.
* **NDCG** - Best when you care about the *order* of relevance (highly relevant items must be first).

## Module 3 - Information retrieval with vector databases

**Vector database** - a database that is suited for semantic search, it’s optimized for storing documents as vectors, and performing retrieval operations on it.

**Why we need vector DB:** When scaling to millions of documents, using regular databases for semantic search fails due to implementation limitations, causing latency and extended resources usage in the system.

### Approximate nearest neighbours algorithms:
**Basic vector retrieval - KNN**
Vectorize all documents -> vectorize prompt -> compute distance from prompt to all documents (in the embedding level) -> return top k

**The issues:** the algorithm scales terribly with linear growth of compression operations with as functions of DB size

**What is ANN:**
* Faster algorithm than KNN
* **HNSW (Hierarchical Navigable Small World):** Currently the most popular algorithm. Creates a multi-layered graph where higher layers allow long jumps (fast traversal) and lower layers allow fine-grained local search. High recall and speed, but consumes more RAM.
* **IVF (Inverted File Index):** Clusters vectors into centroids. During search, it only checks the closest clusters. Efficient for massive datasets and disk-based search, but requires a "training" step to build clusters.

### Vectors databases
This section is about the actual tools used for vector databases, that store the vectors and search using ANN algorithms.

**Popular Databases (2024/2025):**
* **Weaviate:** Open-source, hybrid search (keyword + vector) out of the box, modular (modules for text2vec).
* **Pinecone:** Fully managed, serverless option, very popular for ease of use.
* **Milvus:** Open-source, highly scalable, designed for billions of vectors (Go-based).
* **Qdrant:** Open-source, Rust-based, high performance and low resource usage.
* **pgvector:** Extension for PostgreSQL. Great if you already use Postgres and want to keep data in one place, though typically slower than specialized vector DBs at massive scale.

#### DB operations:
* **Set up the database (User responsibility)** - Establishes the infrastructure and storage architecture optimized for high-dimensional vector operations, which differs significantly from traditional relational databases
* **Load documents (User responsibility)** - Ingests raw text/data that will be converted into searchable vector representations; this is the source material for all subsequent vectorization steps
* **Create sparse vectors for Keyword Search (Database system, often automatic)** - Generates high-dimensional vectors where most values are zero, representing term frequencies or TF-IDF scores; these enable exact keyword matching and traditional lexical search capabilities
* **Create dense embedding vectors for Semantic Search (Database system, often automatic)** - Uses neural networks (like transformers) to convert text into compact, information-rich vectors where every dimension has meaningful values; these capture semantic meaning and context rather than just literal word matches
* **Create the index (e.g., HNSW) (Database system, typically automatic)** - Builds a specialized data structure that enables Approximate Nearest Neighbor (ANN) search, dramatically reducing search time from linear to logarithmic complexity by creating navigable connections between similar vectors
* **Ready to run searches (User can now perform searches)** - The system can now perform hybrid searches combining both keyword precision and semantic understanding, leveraging the index for fast retrieval across potentially millions of vectors

### Chunking
Chunking is the practice of breaking longer text documents into smaller text chunks before vectorization.

**Three main reasons for chunking:**
1. Many embedding models have text length limits
2. Improves search relevancy metrics for retrieval
3. Ensures only most relevant text is sent to the LLM

#### Problems Without Chunking
* **Poor vector representation:** Entire book compressed into single vector averages meaning across all content
* **Weak search relevance:** Vectors can't capture specific topics from particular chapters/pages
* **Context window issues:** Retrieving entire books quickly fills LLM's context window

#### Chunk Size Considerations
* **Chunks That Are Too Large (Chapter Level)**
    * Same problems as whole documents
    * Can't capture nuanced meaning
    * Rapidly fills LLM context window
* **Chunks That Are Too Small (Word/Sentence Level)**
    * Loses context from surrounding text
    * Diminishes search relevance
    * May be too fine-grained for meaningful retrieval

#### Chunking Strategies
1. **Fixed-Size Chunking**
    * **Method:** Every chunk is same predetermined size (e.g., 250 characters)
    * **Process:** Sequential splitting: chars 1-250, 251-500, 501-750, etc.
    * **Problem:** Splits may occur mid-word or separate cohesive thoughts
2. **Overlapping Chunks**
    * **Solution:** Chunks overlap by percentage (typically 10%)
    * **Example:** 250-char chunks with 25-char overlap: 1-250, 226-475, 451-700
    * **Benefits:**
        * Minimizes words cut off from context
        * Edge words appear in multiple chunks
        * Increases odds of relevant context preservation
    * **Trade-off:** More vectors with redundant information
3. **Recursive Character Text Splitting**
    * **Method:** Split on specific characters (e.g., newline between paragraphs)
    * **Advantages:**
        * Accounts for document structure
        * Keeps related concepts together
        * Variable chunk sizes based on natural breaks
    * **Consideration:** May create very large or small chunks
4. **Document-Type Specific Splitting**
    * HTML: Split on paragraph or header tags
    * Python code: Split on function definitions
    * Text documents: Split on newline characters

#### Implementation and Metadata
* **Tools available:** External libraries can handle chunking automatically (e.g., **LangChain TextSplitters**, **LlamaIndex NodeParsers**)
* **Metadata inheritance:** Chunks inherit source document metadata plus location information

#### Recommendations
* **Starting point:** Fixed-size chunks of ~500 characters with 50-100 character overlap
* **Benefits:** Increased search relevancy and optimized LLM context window usage
* **Advanced techniques:** Available for specific use cases requiring more sophisticated approaches

#### Advanced chunking techniques:
**Basic Chunking Problems** - Fixed size and recursive character splitting can break context mid-sentence, losing meaning (e.g., splitting "That night she dreamed...that she was finally an Olympic champion" could make it seem she's already a champion rather than dreaming of it)

1. **Semantic Chunking**
    * Places sentences together in chunks based on similar meaning
    * Algorithm moves through documents one sentence at a time
    * Decides if each sentence is similar enough to previous sentences to belong in same chunk
    * Both current chunk contents and following sentence are vectorized
    * If vectors are below threshold distance, sentences are added to same chunk
    * Process continues until growing chunk becomes too different from next sentence
    * Creates variably sized chunks that follow the author's train of thought
    * Handles conceptual tangents within paragraphs and ideas spanning multiple paragraphs
    * Computationally expensive due to repeated vector calculations for every sentence
    * Often produces high-quality retrieval with good precision and recall metrics
2. **Context-Aware Chunking**
    * Uses LLM to add context to every single chunk
    * LLM creates chunks and adds summary text explaining context within broader document
    * Example: chunk with list of supporters gets explanatory text about its context in the blog post
    * Added context helps with both vectorization (search relevancy) and retrieval (LLM understanding)
    * Requires computationally expensive pre-processing
    * Benefits include more relevant searches with no impact on search speed

### Query Parsing in RAG Systems
**Overview and Purpose** is a critical step in production RAG systems that cleans up and optimizes user-submitted prompts before they're sent to the vector database. This is necessary because:
* Users interact with RAG systems conversationally, as if chatting with another person
* Human-written prompts make poor search queries in their raw form
* The retriever needs to parse prompts to identify intent and transform them for optimal retrieval

#### Basic Query Rewriting (Most Common Approach)
**Process:**
* Uses an LLM to rewrite the user's query before submitting to the retriever
* Most widely used and simplest solution for messy prompts

**Example Implementation:** For a medical knowledge base RAG system, the query rewriter prompt includes instructions to:
* Clarify ambiguous phrases
* Use medical terminology where applicable
* Add synonyms to increase matching odds
* Remove unnecessary or distracting information

**Real Example:**
* **Original user prompt:** "I was out walking my dog, a beautiful black lab named Poppy, when she raced away from me and yanked on her leash hard while I was holding it. Three days later, my shoulder is still numb and my fingers are all pins and needles. What's going on?"
* **Rewritten prompt:** "Experienced a sudden forceful pull on the shoulder resulting in persistent shoulder numbness and finger numbness for three days. What are the potential causes or diagnoses such as neuropathy or nerve impingement?"

**Benefits:**
* Removes unnecessary information (dog's name, breed, etc.)
* Clarifies ambiguity
* Uses medical terminology
* Substantial benefits that justify the additional LLM call cost

### Advanced Query Parsing Techniques

#### Named Entity Recognition (NER)
* **Purpose:** Recognizes categories of information in queries (places, people, dates, fictional characters, etc.)
* **Example with Gliner Model:**
    * Input text with instructions to identify: people, books, locations, dates, actors, characters
    * Model analyzes and returns labeled query with identified entity categories
* **Very efficient model that can run on every query**
* **Adds slight latency but significantly improves retrieval quality**
* **Applications:**
    * Informs vector search performed by retriever
    * Enhances metadata filtering in the process

#### Hypothetical Document Embeddings (HyDE)
* **Concept:** Refines search queries by generating hypothetical ideal result documents
* **Process:**
    * LLM generates a hypothetical document that would be the perfect answer to the query
    * This hypothetical document is embedded into vector representation
    * The vector of the hypothetical document is used for the actual search
* **Example:**
    * For the medical shoulder/hand numbness query
    * LLM generates hypothetical document about shoulder and hand numbness from rapid pulls
    * That hypothetical document's embedding is used for search
* **Advantages:**
    * Helps retriever understand both query intent and what quality results look like
    * Changes retriever task from matching dissimilar texts (questions vs documents) to matching similar texts (document vs document)
    * Provides performance improvements in practice
* **Trade-offs:**
    * Adds latency to search process
    * Requires computational resources for the document generation LLM

### Cross-encoders and ColBERT

**Bi-Encoder** - documents and prompts are embedded separately
* Documents are encoded ahead of time, and prompt in real time, which speeds to search process

**Cross encoder** - a method that provides a better scoring results than the bi-encoder
* Prompt and document are concatenated and then embedded together in the cross-encoder that produces a score between 0 to 1 that grades the similarity/relevances of the prompt with the document
* The method is best for systems that does not needs high latency or has small documents database
* The method is not suited for realtime use cases, however it can improve other methods with the insights of prompt-document relationship it finds

**ColBERT (Contextualized Late Interaction over BERT)**
* Generate documents vectors ahead of time like bi-encoders but also captures deep text interactions like in cross encoder
* Each token gets vector - rather than generating one vector for each prompt and document, each token is vectorized. This operation is done for both document and prompt.
* **Scoring** - each prompt vector tries to find its most similar document vector
* The BERT model is generating embedding vectors for each token, the method then tries to find the cross similarity between the prompts embedding vectors and documents embedding vectors. For each query token, it computes the similarity (dot product) with every document token and takes the maximum similarity score (MaxSim). This means each query token gets matched to its most similar document token, capturing the best possible alignment. The final relevance score between the query and document is the sum of all these maximum similarities across all query tokens.

### Reranking
In the retrieval world, the best approach for accurate retrieval is combining multiple search techniques together.

**Reranking** is a post-retrival process that takes the set of documents returned from less accurate retrieval techniques (i.e. BM25 and semantic search) and uses more accurate methods (cross encoder, ColBERT etc) to order them by relevancy. For example if the quick process returned the most n relevant documents, we can use the reranking process in order to modify order of documents or even take the best k documents where k < n.

**Top Rerankers (2024/2025):**
* **Cohere Rerank:** One of the best performing commercial rerankers via API.
* **BGE-Reranker:** Excellent open-source reranker from BAAI.

We can view this process as mini-RAG on a much smaller database where only the most relevant documents are already chosen.

## Module 4 - LLMs and Text Generation

### LLM sampling strategies

**Temperature** - a variable that controls the “creativity” of the model, by changing the probability distribution of the vocabulary. Lower temperature will assign higher probability to certain tokens, while higher value will spread the probability function. The temperature does not change the ordering of the tokens, only the probability of choosing them.

A clarification about temperature, the model chooses from the set of tokens with probability function determined by each token probability (all summed to 1) so in practice even with temp = 0 (it’s not true zero) the model still samples, with higher temperature values the probability function modified yet, the sampling operation is kept.

The temperature modifies the probability function by:
1. Apply temperature scaling: `scaled_logits = logits / temperature`
2. Convert to probabilities with softmax: `probabilities = softmax(scaled_logits)`

This is the reason T = 0 is not possible -> get smaller values like 0.0001.
**Reminder:** the logits are the score the model gives each token before the softmax (that normalises them between 0-1 and sums all to 1).

**Top-k:** instead of sampling from all vocabulary for next token, the top-k parameter limits the LLM to choose from the K highest ranked tokens.

**Top-p:** similar to top-k but limits the LLM to choose from set of tokens that their accumulated probability equal or lower than p value.
* Top-p is stronger than top-k as it ensures the LLM to choose from the most certain tokens, rather than from K tokens closer to the highest ranked token.

**Repetition penalties** - penalises the LLM from using the same word over and over.

**Logit biases** - allow direct manipulation of tokens probability by adding or subtracting values from the model raw calculated probabilities.

### Choosing the correct LLM:
* Model size
* Cost
* Context window
* Latency and speed
* Training cutoff date

### LLM quality metrics - using benchmarks
* Automated - evaluated using code(MMLU)
* Human scoring
* LLM as a judge

### What we need to consider when choosing LLM:
* **Benchmark quality** - we need to understand what is the task we want to solve and choose the LLM that has best results on benchmarks related to this task
* **Benchmark difficulty** - help to distinguish between high and low performing LLMs. This is crucial, if all LLM scores with high results on the bench mark we can not assess the LLM related to our task. It relates to our consideration that we need to build a golden dataset with hard tasks, such that we could do this bench marking in order to choose a model.
* **Benchmark Reproducibility** - the result that the LLM provider publishes can be reproduced.
* **Benchmark align** - calibrated to real world performance

### Prompt engineering methods:
* **In-context learning** - adding examples to prompts that instruct the LLM about the output.
* **Few-shot learning** - adding many examples in the prompt
* **One-shot learning** - adding single example

In the context of examples we can have a DB that includes good examples instead of adding a lot of examples in the prompt itself.
For our use case - we can use the reasoning the LLM does for the chunking for building a RAG that can provide examples.

**Encouraging reasoning:** encourage the LLM to reason through prompt step by step - this is different from chain of thoughts as we provide the LLM a “place” to think of the solution, and not encourage it to create a plan. BTW this is the thinking we know from LLMs.

**Chain of thoughts** - encouraging the LLM to first create a set of actions then follow them.

When using thinking models we need to be careful to not use methods from regular prompt engineering methods such as:
* COT
* Context learning
* Structured actions
The reason is that these reasoning steps are being trained to do this sort of thing on their own, and will try to please us by mixing the hard constraints into their thinking process.

### Context window preserving methods:
The context window will be filled quickly if we don’t manage it.
* **Massage pruning:** removing last massages from the LLM (keeping N massages)
* **LLM summary:** using another LLM to summarise the message's context, keeping only the keypoints.
* **Reasoning tokens:** we would like to remove the reasoning tokens and keep only response

### Handling Hallucinations:
The HARD truth: there is no absolute solution for LLM hallucinations, in order to reduce them we need validations, in multiple steps of the application.

One of the best options for detecting hallucinations is an external source of truth that can be used to validate the LLM response.

#### Hallucinations detections methods:
* **Self-consistency:** repeatedly generate responses to the same prompt and confirm consistency between responses (using another LLM)
    * The issues with this method is it costly and unreliable
* A better approach is to use retrieved information and instruct the LLM to ONLY base it’s answers on the retrieved information.
* A method to increase the reliability the LLM is only using the resources is to ask it to cite the resources it used in the answer itself.
    * A risk in this approach is the LLM can hallucinate the citations.

### Evaluating citation quality in LLMs
**ALCE benchmark for evaluation** - a system that evaluates RAG pipelines using pre-assembled knowledge base and sample questions.
The system evaluates the LLM response based on the following key metrics:
* **Fluency** - how clear and well written the output is
* **Correctness** - how semanticly accurate the output is
* **Citation quality** - how well do citations align with correct sources

### Evaluating LLM performances
When evaluating LLM in complex pipeline, specifically one that contains retriever, we need to evaluate the specific role of the LLM in the pipeline which is to use the retrieved data in generated response, this means that if the retriever has flows and it retrieves wrong/inaccurate information, and the LLM generated wrong response based on the context retrieved, this is not LLM failure, but retriever failure.

Most LLM evaluation metrics involve other LLM for judgment.

**Popular Frameworks (2024/2025):**
* **Ragas:** A framework that measures component-wise performance using metrics like **Faithfulness** (is answer derived from context?), **Answer Relevancy** (does answer match question?), and **Context Precision** (is retrieved context relevant?).
* **TruLens:** Uses "feedback functions" to evaluate the "RAG Triad": Context Relevance, Groundedness, and Answer Relevance.
* **DeepEval:** A unit testing framework for LLMs that enables TDD (Test Driven Development) for RAG.

**Example Scores (Ragas):**

* **Response relevancy** - measure where response is relevant to user prompt regardless of accuracy
    * Evaluator LLM generated several new “sample prompts” that could have leas to the same response
    * Embed original and sample prompts to vectors and compute similarity
    * Average similarity scores for final relevancy measures
    * **Example:** the answer is "Paris is the capital of France", the LLM first generates different questions that the answer could have been suited to,"What is the capital of France?", "Which city is France's capital?" etc, then the embedding model embeds all the questions and measures their distance from the original question, and the last step is averaging all similarities. This calculation helps us to understand if the system provided answers that closely relate to questions that can be asked about it.
* **Faithfulness** - measure whether response in consistent with retrieved information
    * LLM identifies all factual claims in response
    * More LLM calls to determine if claims are factually supported by retrieved information
    * Percentage of supported claims is the faithfulness

**System wise metrics** - since RAG is part of agentic/LLM based system, we can use personal metrics (such as accepted line of codes in coding assistance, like/dislikes etc) to measure our system, however it’s very important to isolate the exact issue (either LLM generation or retrieval)

### Agentic RAG

#### What Are Agentic Workflows?
* Using multiple LLMs throughout your RAG system, each responsible for a single step
* **Examples:** query expansion, prompt rewriting, citation generation
* Powerful way to improve RAG performance as system matures
* **GraphRAG:** An advanced pattern where a Knowledge Graph is used alongside (or instead of) vector search to capture structured relationships between entities, improving "global" understanding.

#### Key Differences from Traditional LLM Use
* **Tasks broken into series of steps:** Each step completed by different LLM call
* **LLMs given access to tools:** Code interpreter, web browser, vector database

#### Example Agentic RAG System
* **Router LLM:** Determines if prompt requires vector database retrieval (outputs yes/no)
* **Retrieval step:** Fetches documents if needed, or skips if not
* **Evaluator LLM:** Checks if retrieved documents are sufficient; may trigger additional retrievals
* **Generator LLM:** Creates response using augmented prompt
* **Citation LLM:** Adds citations to final response

### RAG vs finetuning

#### Current Consensus:
* **RAG:** Best for knowledge injection (adding new information)
* **Fine-tuning:** Best for domain adaptation (specializing in tasks/domains)

#### When to Use RAG
* LLM needs access to new information
* Inject information into prompt
* Off-the-shelf LLM can incorporate new info in response

#### When to Use Fine-Tuning
* Want LLM to specialize in certain task or domain
* Handling one discrete task (e.g., routing prompts in RAG system)
* Responding to specific type of prompt only

#### Using Both Together
* RAG and fine-tuning can be combined
* **Example:** Fine-tune model to better incorporate retrieved information into responses
* Helps model specialize in its role within the RAG system

## Module 5: RAG Systems in Production

### What makes production challanging
* **More traffic** - more users are using the system
* **More requests** - complementary to more traffic
* **Scaling** - more resources are needed
* **Unpredictability of prompts** - users are very creative, predicting any user query is imposible
* **Real world data is mess** - fragmented, messy, missing metadata
* **Security and privacy**
* **The “eat rocks” use case** - a user is asking how much rocks he should eat a day, and the system is not responding with “humans not eat rocks” but provide an answer with the number of rocks he should eat, this use case showcases production errors that can affect the reputation of the company.

### Implementing RAG evaluation strategies
The first step in handling AI production challenges (specifically agents and RAG) is building robust observability systems. These systems should include the following components:

#### Key metrics
1. **Software performance metrics** - agent/RAG is in its core a software system, and as one, we need to observe its performance. For example: latency, throughput, memory and compute usage.
2. **Quality metrics** - measure user satisfaction, and system output quality.

#### How to track
* **Aggregate statistics** - track trends and identify regression over time
* **Detailed logs** - trace individual prompts through your pipeline, this process helps to understand the journey the data traverse in the system and to locate the failure points. **LangSmith** and **Arize Phoenix** are popular tools for this trace visualization.
* **Experiments** - A/B test changes and run secure experiments. In case we want to change a component in the system (system prompt, provider, tuning the retrieval params, etc) we need to first test it in a secure environment. Another option is to use A/B testing with users in production.

#### Scope vs evaluator
* **Scope** - whether an evaluation is targeting a component of the RAG system, or the overall system.
* **Evaluator** - how the evaluation is done, code-base, another LLM, human feedback, hybrid.
