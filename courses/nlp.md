
# Natural Language Processing Specialization


# [Certificate](https://www.coursera.org/account/accomplishments/specialization/certificate/ECZE48SJP9B6)


## Course 1 NLP with Classification and Vector Spaces
### Week 1  logistic regression and its application to sentiment analysis of text data
**Logistic Regression:**

* **Sigmoid Function:** Outputs probability (0 to 1). Formula: $\frac{1}{1 + e^{-\theta ^Tx^{(i)}}}$.
* **Training:**
    * Initialize $\theta$.
    * Iteratively:
        * Compute gradient.
        * Update $\theta$: $\theta := \theta - \alpha \nabla J(\theta)$.
        * Calculate cost.
    * Continue until cost converges.
* **Cost Function:**
    * Formula: $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log h(x^{(i)},\theta) + (1-y^{(i)})\log(1-h(x^{(i)},\theta))]$.
    * Derived from maximizing the log-likelihood.
    * Minimizing cost $\Leftrightarrow$ maximizing likelihood.
* **Gradient Descent:**
    * General form: $\theta := \theta - \alpha \nabla J(\theta)$.
    * Logistic Regression Gradient:
        * Non-vectorized: $\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$.
        * Vectorized: $\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)$.
    * Derivative of Sigmoid: $h'(x) = h(x) (1 - h(x))$.
* **Testing:**
    * Use validation/test set.
    * Prediction: $\geq 0.5 \rightarrow$ positive, $< 0.5 \rightarrow$ negative.
    * Data Split: Train, Validation, Test (e.g., 80/10/10).
    * Accuracy: $\frac{\text{Number of correct predictions}}{m}$.

**Supervised ML & Sentiment Analysis:**

* Input ($X$) $\rightarrow$ Prediction ($\hat{y}$).
* Compare $\hat{y}$ with True ($Y$) $\rightarrow$ Cost.
* Cost updates parameters ($\theta$).
* Sentiment Analysis: Text $\rightarrow$ Features $\rightarrow$ Logistic Regression $\rightarrow$ Sentiment (Positive/Negative).

**Vocabulary & Feature Extraction (Basic):**

* Represent text as a vector of size $V$ (vocabulary size).
* 1 if word present, 0 otherwise.
* Sparse vectors, high dimensionality.

**Feature Extraction with Frequencies:**

* Create `freqs` dictionary: `(word, class) -> count`.
* Encode tweet as a low-dimensional vector (e.g., [bias, positive\_freq, negative\_freq]).
* Calculate positive and negative frequencies for each word in the tweet using `freqs`.

**Preprocessing:**

1.  Eliminate handles and URLs.
2.  Tokenize into words.
3.  Remove stop words (e.g., "and", "is", "a").
4.  Stemming (e.g., "dancer", "dancing" $\rightarrow$ "danc"). Use Porter stemmer.
5.  Convert to lowercase.

**Putting it All Together:**

* Text input $\rightarrow$ Preprocessing $\rightarrow$ Feature Extraction ($\rightarrow$ $X$ matrix) $\rightarrow$ Logistic Regression (Training & Prediction).

### Week 2 Naïve Bayes

**Naïve Bayes:**

* **Assumptions:**
    * **Independence Assumption:** Assumes all features (e.g., words) are independent of each other, which is often not true in reality (e.g., "sunny" and "hot").
    * **Word Frequencies:** Affected by the frequency of words in the training data.
* **Relative Frequencies in Corpus:**
    * Real-world data (e.g., Twitter) often has imbalanced class distributions (more positive than negative).
    * "Clean" datasets might be artificially balanced.

**Error Analysis:**

* Common reasons for misclassification:
    * Removing punctuation (can lose sentiment information).
    * Removing words (can lose crucial context).
    * Word order (Naïve Bayes ignores order).
    * Adversarial attacks (sarcasm, irony, euphemisms are hard to capture).

**Probability and Bayes’ Rule:**

* **Probability:** $P(\text{event}) = \frac{\text{Count of event}}{\text{Total count of all events}}$.
* Sum of all probabilities = 1.
* **Probability of two events (intersection):** $P(A \cap B) = \frac{\text{Count of overlap}}{\text{Total count}}$.

**Bayes' Rule:**

* Conditional Probability reduces search space. $P(A|B) = \frac{P(A \cap B)}{P(B)}$.
* Derivation:
    * $P(\text{Positive} | \text{happy}) = \frac{P(\text{Positive} \cap \text{happy})}{P(\text{happy})}$
    * $P(\text{happy} | \text{Positive}) = \frac{P(\text{happy} \cap \text{Positive})}{P(\text{Positive})}$
    * Substituting: $P(\text{Positive} | \text{happy}) = P(\text{happy} | \text{Positive}) \times \frac{P(\text{Positive})}{P(\text{happy})}$
* **Bayes' Rule Formula:** $P(X|Y) = \frac{P(Y|X)P(X)}{P(Y)}$.

**Naïve Bayes Introduction:**

* Classifier building starts with conditional probabilities from training data.
* **Likelihood Score:** $\frac{P(\text{document}|Positive})}{P(\text{document}|Negative})} = \prod_{i=1}^{n} \frac{P(w_i|Positive})}{P(w_i|Negative})}$.
* Score > 1 $\rightarrow$ Positive, Score < 1 $\rightarrow$ Negative.

**Laplacian Smoothing:**

* Problem: Zero probability if a word doesn't appear in a class.
* **Smoothing Formula:** $P(w_i | class) = \frac{freq(w_i, class) + 1}{(N_{class} + V)}$.
    * $N_{class}$: Total word frequency in the class.
    * $V$: Number of unique words (vocabulary size).

**Log Likelihood:**

* Use ratios for scoring. Higher ratio $\rightarrow$ more positive.
* Inference: $\frac{P(pos)}{P(neg)} \prod_{i=1}^{m} \frac{P(w_i|pos)}{P(w_i|neg)} > 1$.
* Logarithm for numerical stability with large $m$: $\log \frac{P(pos)}{P(neg)} + \sum_{i=1}^{n} \log \frac{P(w_i|pos)}{P(w_i|neg)}$.
    * First term: Log Prior.
    * Second term: Log Likelihood.
* $\lambda(w) = \log \frac{P(w_i|pos)}{P(w_i|neg)}$ dictionary for efficient inference.

**Log Likelihood Part 2 (Inference):**

* Score = $logprior + \sum_{i \in \text{document}} \lambda(i)$.
* Score > 0 $\rightarrow$ Positive, Score < 0 $\rightarrow$ Negative.

**Training Naïve Bayes:**

1.  Annotated dataset (positive/negative tweets).
2.  Preprocessing: `process_tweet()`
    * Lowercase
    * Remove punctuation, URLs, names
    * Remove stop words
    * Stemming
    * Tokenization
3.  Compute `freq(w, class)` (word counts per class).
4.  Calculate $P(w_i|pos)$, $P(w_i|neg)$ (probabilities with smoothing).
5.  Calculate $\lambda(w) = \log \frac{P(w_i|pos)}{P(w_i|neg)}$.
6.  Calculate $logprior = \log \frac{D_{pos}}{D_{neg}}$ (log of positive to negative document ratio).

**Testing Naïve Bayes:**

* Use $\lambda$ dictionary and $logprior$ to predict sentiment of new text.
* Score > 0 $\rightarrow$ Positive, Score < 0 $\rightarrow$ Negative.
* Example shows if $logprior = 0$, classification depends solely on the sum of $\lambda$ values for the words in the document.

**Applications of Naive Bayes:**

* Author identification
* Spam filtering
* Information retrieval
* Word disambiguation
* Simple baseline, fast.

### Week 3  Vector Space Models

**Vector Space Models:**

* **Fundamental in NLP:** Represent words, documents, tweets as vectors.
* **Applications:** Information extraction, machine translation, chatbots, relationship identification.
* **Word Relationships:** Vectors can show semantic relationships.
    * Firth's quote: "You shall know a word by the company it keeps."
    * Meaning from neighboring words.
    * Clustering vectors: Adjectives near adjectives, nouns near nouns, etc.
    * Synonyms & antonyms tend to be close due to similar neighboring words.

**Word by Word and Word by Doc:**

* **Word by Word Design:**
    * Matrix: Rows & columns = vocabulary.
    * Count co-occurrence of words within a window ($K$).
    * Example: Word "data" vector $v=[2,1,1,0]$ (counts with other words).
* **Word by Document Design:**
    * Matrix: Rows = words, Columns = documents.
    * Values = word frequency in each document.
    * Example: "Entertainment" category vector $v=[500, 7000]$ (word counts).
    * Can compare categories by plotting vectors.
    * Angle between vectors (later): Measures similarity.

**Euclidean Distance:**

* Distance between points $A$ and $B$: $d(B, A) = \sqrt{(B_1 - A_1)^2 + (B_2 - A_2)^2}$.
* Generalization to n-dimensional vectors: $d(\vec{v}, \vec{w}) = \sqrt{\sum_{i=1}^{n} (v_i - w_i)^2}$.
* Example calculation shown.

**Cosine Similarity: Intuition:**

* Problem with Euclidean distance: Not always accurate for semantic similarity, especially with documents of different lengths.
* Example: Food corpus vs. Agriculture corpus (same word proportions, different sizes). Euclidean distance might incorrectly suggest History is closer to Agriculture.
* Cosine similarity focuses on the angle between vectors, capturing proportional similarity.

**Cosine Similarity:**

* **Norm of a vector:** $\|\vec{v}\| = \sqrt{\sum_{i=1}^{n} |v_i|^2}$.
* **Dot product:** $\vec{v} \cdot \vec{w} = \sum_{i=1}^{n} v_i \cdot w_i$.
* **Cosine Similarity Formula:** $\cos(\beta) = \frac{\vec{v} \cdot \vec{w}}{\|\vec{v}\| \|\vec{w}\|}$ (normalized vectors $\hat{v}, \hat{w}$).
* Same vectors ($\beta=0$): $\cos(0) = 1$.
* Orthogonal vectors ($\beta=90$): $\cos(90) = 0$ (dot product is 0).

**Manipulating Words in Vector Spaces:**

* Word vectors can reveal semantic patterns.
* Example: $\vec{Russia} - \vec{USA} + \vec{DC} \approx \vec{Moscow}$.
* Cosine similarity can confirm that the resulting vector is closest to $\vec{Moscow}$.
* Distance and direction between country and capital are relatively consistent.

**Visualization and PCA:**

* **Principal Component Analysis (PCA):** Unsupervised dimensionality reduction for visualization.
* Combines variances across features.
* Example: PCA on data shows "oil & gas" close, "town & city" close.
* Reduces dimensions (e.g., $d>2$ to $d=2$) for plotting.
* Plotting word vectors (with PCA): Similar Part-of-Speech (POS) tags are near.
* Synonyms and antonyms also tend to be close (why?).

**PCA Algorithm:**

* Intuitively: Collapses data onto principal components (directions of highest variance).
* First principal component (2D): Line with most variance.
* **Eigenvector:** Uncorrelated features.
* **Eigenvalue:** Amount of information (variance) in each eigenvector.
* Steps:
    1. Mean normalize data.
    2. Compute covariance matrix.
    3. Compute Singular Value Decomposition (SVD) on covariance matrix: $[U, S, V] = svd(\Sigma)$.
        * $U$: Eigenvectors.
        * $S$: Eigenvalues.
    4. Reduce dimension: Use first $n$ columns of $U$ to get new data: $X_{new} = X \cdot U[:, 0:n]$.

### Week 4 Transforming Word Vectors and Nearest Neighbors using hashing techniques

**Transforming Word Vectors:**

* **Goal:** Learn a "transformation matrix" ($R$) to translate word vectors from one language to another (e.g., English to French).
* **Mechanism:** Multiply an English word vector ($X$) by $R$ to approximate its French equivalent ($Y$): $XR \approx Y$.
* **Learning $R$:**
    * Initialize $R$.
    * Iteratively minimize the Frobenius norm of the difference: $Loss = ||XR - Y||_F^2$.
    * Update $R$ using gradient descent: $R = R - \alpha * g$, where $g$ is the gradient of the loss.
* **Frobenius Norm:** $\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2}$ (square root of the sum of squares of all elements).
* **Example:** Transforming "cat" (English vector) using $R$ should result in a vector close to "chat" (French vector) based on cosine similarity.

**K-Nearest Neighbors (KNN) and Hashing:**

* **Goal:** Find the most similar vectors (nearest neighbors) to a query vector.
* **Challenge:** Searching all vectors can be computationally expensive, especially in high dimensions.
* **Hashing:** Used for faster lookups by grouping similar items into "buckets."
* **Hash Function:** Maps data of arbitrary size to a fixed-size hash value.
* **Basic Hash Table:** Modulo operation to assign values to buckets: `hash_value = int(value_l) % n_buckets`.
* **Lookup:** Only compare the query vector to items within its hash bucket.

**Locality Sensitive Hashing (LSH):**

* **Goal:** Hash similar inputs into the *same* buckets with high probability.
* **Mechanism:** Uses planes (or lines in 2D) to divide the vector space.
* **Projection (Dot Product):** The sign of the dot product between a vector and a plane's normal vector indicates which side of the plane the vector lies on.
* **Multiple Planes:** Combine the results of multiple projections to create a single hash value.
* **Hash Value Calculation:** $hash\_value = \sum_{i=1}^{H} 2^{i} \times h_{i}$, where $h_i$ is 1 if the point is on one side of the $i$-th plane and 0 otherwise.

**Approximate Nearest Neighbors:**

* **Goal:** Find an *approximation* of the nearest neighbors, trading accuracy for efficiency.
* **LSH Application:** By hashing vectors using multiple sets of random planes, you can retrieve candidate neighbors from the buckets the query vector falls into.
* **Multiple Lookups:** Performing the hashing and lookup multiple times increases the probability of finding the true nearest neighbors.

**Searching Documents:**

* **Document as a Vector:** Summing the word vectors of the words in a document can represent the document.
* **Summary of Concepts:** Word vectors, transformation matrices, KNN, hash tables, hash functions, LSH, approximate nearest neighbors.

## Course 2 NLP with Probabilistic Models
### Week 1 Autocorrect and Minimum Edit Distance

**Overview:**

* Autocorrect is a common feature in text input.
* Minimum Edit Distance: Measures the minimum number of edits to transform one word into another.
* Introduces Dynamic Programming as a key technique for solving optimization problems like minimum edit distance.

**Autocorrect Implementation Steps:**

1.  **Identify Misspelled Word:** Check if the word exists in the vocabulary. If not, it's likely a typo.
2.  **Find Strings n Edit Distance Away:** Generate candidate words that are within a certain edit distance (n) from the misspelled word. These could initially be random strings.
3.  **Filter Candidates:** Keep only the candidate strings that are actual words found in the vocabulary.
4.  **Calculate Word Probabilities:** Choose the most likely correct word based on its frequency of occurrence in a given context (for this week, just word frequencies). More sophisticated autocorrect can consider the probability of word sequences (n-grams).

**Build The Model (Details):**

* **Step 1 (Identify):** Straightforward vocabulary lookup.
* **Step 2 (Find Edit Distance):** Generating variations by inserting, deleting, or replacing letters.
* **Step 3 (Filter):** Checking generated words against a known vocabulary.
* **Step 4 (Probabilities):** Using word counts to estimate probabilities (e.g., $P(word) = \frac{count(word)}{\text{total word count}}$).

**Minimum Edit Distance:**

* **Purpose:** Evaluate the similarity between two strings.
* **Edits:**
    * Insert (add a letter).
    * Delete (remove a letter).
    * Replace (change one letter to another).
* **Cost:** Each edit operation has a cost (usually 1 for Levenshtein distance). The goal is to find the minimum total cost to transform one string to another.
* Calculating edit distance manually becomes difficult for longer strings.

**Minimum Edit Distance Algorithm:**

* Uses a table (matrix) to store the minimum edit distances between prefixes of the source and target words.
* **Initialization:** The first row and column are initialized with the cost of deletions and insertions respectively.
* **Table Population:** Each cell $D[i,j]$ represents the minimum edit distance between the first $i$ characters of the source word and the first $j$ characters of the target word.
* **Recurrence Relations:** To calculate $D[i,j]$, consider three possibilities:
    * **Deletion:** $D[i,j] = D[i-1, j] + del\_cost$ (delete a character from the source).
    * **Insertion:** $D[i,j] = D[i, j-1] + ins\_cost$ (insert a character into the source).
    * **Replacement/Match:** $D[i,j] = D[i-1, j-1] + rep\_cost$, where $rep\_cost$ is 0 if the characters at the current positions match, and a cost (e.g., 2 or 1) if they need to be replaced.
* The value in the bottom-right cell of the table represents the minimum edit distance between the two full strings.

**Minimum Edit Distance III (Backtrace & Dynamic Programming):**

* **Levenshtein Distance:** A common type of edit distance where each insertion, deletion, or substitution has a cost of 1.
* **Backtrace:** To find the actual sequence of edits, keep track of which previous cell led to the minimum cost at each step (using pointers). This allows reconstructing the path from the top-left to the bottom-right of the table.
* **Dynamic Programming:** The minimum edit distance algorithm is an example of dynamic programming. It solves smaller overlapping subproblems (distances between prefixes) and stores their solutions in the table to avoid redundant calculations when solving larger subproblems. This efficient reuse of results is the core idea of dynamic programming.

### Week 2 Speech Tagging, Markov Chains, Hidden Markov Models, and the Viterbi Algorithm

**Part of Speech Tagging (POS):**

* Assigning a part of speech (e.g., noun, verb) to each word in a text.
* Key concepts: Markov Chains, Hidden Markov Models (HMMs), Viterbi algorithm.
* Applications: Named entity recognition, speech recognition, coreference resolution.
* Uses probabilities of POS tags occurring near each other to determine the most likely sequence.

**Markov Chains:**

* Model the probability of the next state (e.g., next word's POS) based on the current state.
* States ($Q$) represent conditions (e.g., POS tags).
* Transitions between states have probabilities (e.g., probability of a noun following a verb).
* Requires identifying probabilities of POS tags and words.
* **Transition Matrix (A):** $A[i][j] = P(S_j | S_i)$, probability of transitioning from state $S_i$ to $S_j$. The first row often represents the initial distribution of POS tags.

**Hidden Markov Models (HMMs):**

* Extends Markov Chains with **emission probabilities**.
* **Emission Probabilities (Matrix B):** $B[i][k] = P(O_k | S_i)$, probability of observing a specific word ($O_k$) given a hidden state (POS tag $S_i$).
* Uses both transition matrix (A) and emission matrix (B) to determine the POS of words in a sentence.
* Matrices A and B are populated from labeled data by calculating the probabilities of POS tag sequences and POS tags generating specific words.
* Sum of each row in A and B must equal 1.

**Calculating Probabilities (for A & B):**

* Count occurrences and normalize.
* **Transition Probability:** $P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_i)}{\sum_{j=1}^N C(t_{i-1}, t_j)}$ (count of tag $t_i$ following $t_{i-1}$ divided by total occurrences of $t_{i-1}$).
* **Emission Probability:** $P(w_i | t_i) = \frac{C(t_i, w_i)}{C(t_i)}$ (count of word $w_i$ with tag $t_i$ divided by total occurrences of tag $t_i$).
* **Smoothing (using $\epsilon$):** Added to the numerator and denominator to avoid zero probabilities for unseen transitions or emissions.

**Populating Transition Matrix (A):**

* Count occurrences of each tag followed by another.
* Normalize counts to get probabilities.
* Initial state ($\pi$ or orange) represents the probability of a sentence starting with a particular POS tag.

**Populating Emission Matrix (B):**

* Count occurrences of each word with each POS tag.
* Normalize counts to get probabilities.
* Smoothing is applied: $P(w_i | t_i) = \frac{C(t_i, w_i) + \epsilon}{C(t_i) + V \epsilon}$ (where $V$ is vocabulary size).

**The Viterbi Algorithm:**

* Uses dynamic programming to find the most likely sequence of hidden states (POS tags) for a given sequence of observed words.
* **Initialization (Matrix C):** $C[i, 1] = \pi[i] * B[i, cindex(w_1)]$ (probability of the first word having tag $i$).
* **Forward Pass (Matrix C & D):**
    * For each word $j$ and each tag $i$:
        * $C[i,j] = \max_k (C[k, j-1] * A[k, i] * B[i, cindex(w_j)])$ (probability of the sequence ending with word $j$ having tag $i$).
        * $D[i,j] = \text{argmax}_k (C[k, j-1] * A[k, i])$ (stores the index of the previous tag $k$ that led to the maximum probability).
* **Backward Pass (Reconstructing POS sequence):**
    * Find the tag with the highest probability in the last column of C. This is the POS tag for the last word.
    * Use the D matrix to backtrack and retrieve the most likely tag for the preceding words. $t_j = D[t_{j+1}, j+1]$.

**Transfer and Emission Probability Matrices (Recap):**

* **Transition Probability Matrix (A):** $P(\text{next hidden state} | \text{current hidden state})$.
* **Emission Probability Matrix (B):** $P(\text{observed value} | \text{hidden state})$.
* Used together in algorithms like Viterbi to predict the most likely sequence of hidden states.
### Week 3 n-gram

**N-Grams Overview:**

* N-grams model the probability of word sequences.
* Fundamental for tasks like autocorrect and search suggestions.
* Other applications: Machine translation, spam filtering, speech recognition.
* Topics covered: Building N-gram models, handling out-of-vocabulary (OOV) words, smoothing, and model evaluation.

**N-grams and Probabilities:**

* **N-gram:** A sequence of N items (words in language models). Unigram (1), Bigram (2), Trigram (3), etc.
* **Probability of a Unigram:** $P(w) = \frac{C(w)}{m}$ (count of word / total words).
* **Probability of a Bigram:** $P(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$ (count of bigram / count of preceding word).
* **Probability of a Trigram:** $P(w_3 | w_1 w_2) = \frac{C(w_1 w_2 w_3)}{C(w_1 w_2)}$.
* **Probability of an N-gram:** $P(w_N | w_1^{N-1}) = \frac{C(w_1^{N})}{C(w_1^{N-1})}$.

**Sequence Probabilities:**

* Probability of a sentence: $P(w_1 w_2 ... w_n) = P(w_1)P(w_2|w_1)P(w_3|w_1 w_2)...P(w_n|w_1^{n-1})$.
* **Markov Assumption:** Approximates the probability of a word based only on the preceding N-1 words.
    * Bigram: $P(w_n | w_1^{n-1}) \approx P(w_n | w_{n-1})$.
    * N-gram: $P(w_n | w_1^{n-1}) \approx P(w_n | w_{n-N+1}^{n-1})$.
* Sentence probability approximation: $P(w) \approx \prod_{i=1}^{n} P(w_i | w_{i-1})$.

**Starting and Ending Sentences:**

* Use special tokens: `<s>` (start), `</s>` (end).
* Add N-1 `<s>` tokens at the beginning for N-gram models.
* Add one `</s>` token at the end.
* These tokens are included when calculating probabilities.

**The N-gram Language Model:**

* Components: Count matrix, probability matrix.
* **Count Matrix:** Rows = unique (N-1)-grams, Columns = unique words, Values = counts.
* **Probability Matrix:** $P(w_n | w_{n-N+1}) = \frac{C(w_{n-N+1}, w_n)}{C(w_{n-N+1})}$.
* **Language Model Applications:** Sentence probability, next word prediction, generative models.
* **Log Probability:** Used to avoid underflow when multiplying many small probabilities: $log(P(w)) \approx \sum_{i=1}^n log(P(w_i | w_{i-1}))$.
* **Generative Model:** Sample the next word based on the probabilities.

**Language Model Evaluation:**

* **Data Splitting:** Train (80-98%), Validation (1-10%), Test (1-10%).
    * Chronological split for time-sensitive data.
    * Random split otherwise.
* **Perplexity (PP):** Measures how well a language model predicts a sample. Lower perplexity = better model (more human-like).
    * $PP(W) = P(s_1, ..., s_m)^{-\frac{1}{m}} = \sqrt[m]{\prod_{i=1}^m \prod_{j=1}^{s_i} \frac{1}{P(w_j^{(i)} | w_{j-1}^{(i)})}}$.
    * Log Perplexity: $log PP(W) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^{s_i} log_2 P(w_j^{(i)} | w_{j-1}^{(i)})$.

**Out of Vocabulary (OOV) Words:**

* **Vocabulary:** Set of unique words the model knows.
* **Closed Vocabulary:** Only words from a fixed set are encountered/generated.
* **Open Vocabulary:** Encounters words outside the vocabulary.
* **Handling OOVs:**
    1. Create a vocabulary $V$.
    2. Replace words in the corpus not in $V$ with `<UNK>` (unknown token).
    3. Calculate probabilities treating `<UNK>` as a regular word.
* **Creating Vocabulary:**
    * Minimum word frequency.
    * Maximum vocabulary size (based on frequency).
    * Use `<UNK>` sparingly to avoid losing information and hindering generalization.
    * Compare LMs with the same vocabulary size when using perplexity.

**Smoothing:**

* **Problem:** Zero probability for unseen N-grams.
* **Add-1 Smoothing (Laplace):** $P(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$ ($V$ = vocabulary size).
* **Add-k Smoothing:** $P(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + k}{C(w_{n-1}) + k * V}$.
* **Back-off:** If an N-gram is missing, use lower-order (N-1)-gram probabilities (can distort distribution, requires discounting).
    * **"Stupid" Backoff:** Multiply lower-order probability by a constant (e.g., 0.4).
* **Interpolation:** Combine probabilities from different order N-grams with weights ($\lambda_i$) that sum to 1:
    * $\hat{P}(w_n | w_{n-2}w_{n-1}) = \lambda_1 P(w_n | w_{n-2}w_{n-1}) + \lambda_2 P(w_n | w_{n-1}) + \lambda_3 P(w_n)$.

### Week 4 CBOW

**Overview:**

* Word embeddings represent words as dense vectors of numbers.
* Crucial for most NLP applications as they encode semantic meaning.
* Applications: Machine translation, sentiment analysis, question answering, text summarization.
* Topics covered: Concepts of word representations, generating embeddings, text preparation, and the CBOW model.

**Basic Word Representations:**

* **Integers:** Simple but lack inherent meaning or relationships between words.
* **One-Hot Vectors:** Binary vectors with a single '1' representing a word's index in the vocabulary.
    * **Pros:** Simple, no implied ordering.
    * **Cons:** Huge dimensionality (size of vocabulary), encode no semantic meaning.
* **Word Embeddings:** Dense, low-dimensional vectors that capture semantic relationships. Similar words have similar embeddings (close in vector space).

**Why Use Word Embeddings?**

* Capture semantic relationships (e.g., "happy" and "joyful" are close).
* Can represent abstract and concrete properties along different dimensions.
* Lower dimensionality compared to one-hot vectors.

**How to Create Word Embeddings?**

* Need a large text corpus and an embedding method.
* **Context:** Words that appear near a target word provide its meaning.
* **Self-Supervised Learning:** Models learn from the unlabelled corpus itself by predicting words based on their context.
* **Hyperparameters:** Embedding dimension is a key parameter to tune.
* **Embedding Methods:** Classical (word2vec, CBOW, Skip-gram, GloVe, fastText) and Deep Learning/Contextual (BERT, ELMo, GPT-2).

**Word Embedding Methods (Classical):**

* **CBOW (Continuous Bag-of-Words):** Predicts a center word given its surrounding context words.
* **Skip-gram:** Predicts surrounding context words given a center word.
* **GloVe (Global Vectors):** Uses word co-occurrence statistics from the entire corpus.
* **fastText:** Extends Skip-gram by considering subword information (n-grams of characters), better for OOV words.

**Continuous Bag of Words (CBOW) Model:**

* **Goal:** Predict a missing center word based on its context.
* **Sliding Window:** Used to create context-center word pairs from the corpus.
* **Context Size (C):** Number of words before and after the center word used as input.

**Cleaning and Tokenization:**

* Essential preprocessing steps before training.
* Includes lowercasing, removing punctuation, handling special characters, splitting text into words (tokenization).

**Sliding Window in Python:**

* Generates context-center word pairs from a list of words based on the context size C.

**Transforming Words into Vectors (for CBOW Input):**

* Context words are typically represented as one-hot vectors.
* These one-hot context vectors are averaged to create a single input vector ($X$) for the CBOW model.

**Architecture for the CBOW Model:**

* **Input ($X$):** Average of context word vectors.
* **Hidden Layer ($h$):** $z_1 = W_1x + b_1$, $h = ReLU(z_1)$. $W_1$ (V x n), $b_1$ (n x 1), $x$ (V x 1), $z_1$ (n x 1), $h$ (n x 1).
* **Output Layer ($\hat{y}$):** $z_2 = W_2h + b_2$, $\hat{y} = softmax(z_2)$. $W_2$ (n x V), $b_2$ (V x 1), $z_2$ (V x 1), $\hat{y}$ (V x 1) (probability distribution over vocabulary).
* **Prediction:** The word corresponding to the argmax of $\hat{y}$.
* **Batch Input:** Examples stacked as columns in matrices.

**Activation Functions:**

* **ReLU (Rectified Linear Unit):** $ReLU(x) = max(0, x)$. Introduces non-linearity.
* **Softmax:** Transforms a vector into a probability distribution where elements sum to 1. $\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$.

**Training a CBOW Model: Cost Function:**

* **Cross-Entropy Loss:** $J = -\sum_{k=1}^{V} y_k \log \hat{y}_k$, where $y$ is the one-hot vector of the true center word.

**Training a CBOW Model: Forward Propagation:**

* Sequence of calculations from input to prediction: $Z_1 \rightarrow H \rightarrow Z_2 \rightarrow \hat{Y}$.
* **Batch Loss:** $J_{batch} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{V}y_{j}^{(i)}\log{\hat{y}}_{j}^{(i)}$.

**Training a CBOW Model: Backpropagation and Gradient Descent:**

* **Backpropagation:** Calculates gradients of the loss with respect to weights ($W_1, W_2$) and biases ($b_1, b_2$).
* **Gradient Descent:** Updates weights and biases iteratively to minimize the loss: $W := W - \alpha \frac{\partial J}{\partial W}$, $b := b - \alpha \frac{\partial J}{\partial b}$ (where $\alpha$ is the learning rate).

**Extracting Word Embedding Vectors:**

* **Option 1:** Use the columns of the first weight matrix $W_1$.
* **Option 2:** Use the rows of the second weight matrix $W_2$.
* **Option 3:** Average the embeddings from $W_1$ and $W_2$.

**Evaluating Word Embeddings:**

* **Intrinsic Evaluation:** Tests the relationships between words directly (e.g., analogies, clustering, visualization).
    * Semantic analogies (France:Paris :: Italy:?).
    * Syntactic analogies (seen:saw :: been:?).
* **Extrinsic Evaluation:** Evaluates the usefulness of embeddings on downstream NLP tasks (e.g., NER, POS tagging).
    * **Pros:** Evaluates actual usefulness.
    * **Cons:** Time-consuming, harder to troubleshoot.


## Course 3 NLP with sequence models
### Week 1 Neural Networks and Recurrent Neural Networks (RNNs) for Sentiment Analysis:

**Neural Networks for Sentiment Analysis:**

  * NNs can capture sentiment in more complex sentences compared to Naive Bayes or Logistic Regression (e.g., handling negation).
  * Input tweet can be represented as a sequence of word embeddings (padded to the length of the longest tweet).
  * Multi-layer NNs (Dense layers with activation functions like ReLU) can be used for classification (e.g., positive, neutral, negative sentiment).
  * Training is more efficient by processing batches of tweets in parallel using matrix operations.

**Dense Layers and ReLU:**

  * **Dense Layer:** Computes the inner product of the input vector and a weight matrix, adding a bias.
  * **ReLU (Rectified Linear Unit):** Activation function $max(0, x)$ applied element-wise, introducing non-linearity.

**Embedding and Mean Layers:**

  * **Embedding Layer:** Learns word embeddings for each word in the vocabulary, mapping one-hot vectors to dense vectors.
  * **Mean Layer:** Calculates the average of the word embeddings in a sequence (e.g., a tweet), resulting in a single vector representation of the input. This layer has no trainable parameters.

**Traditional Language Models (N-grams):**

  * Predict the probability of word sequences.
  * Larger N-grams can capture long-range dependencies but require significant memory.
  * Probability of a sentence is the product of conditional probabilities of words.

**Recurrent Neural Networks (RNNs):**

  * Address the limitations of N-grams by maintaining a hidden state that captures information from the entire input sequence.
  * Can theoretically learn long-range dependencies.
  * As the RNN processes a sequence, the influence of earlier words in the hidden state might weaken over long sequences.
  * Parameter sharing across time steps makes RNNs efficient.

**Applications of RNNs:**

  * One to One (e.g., winner prediction).
  * One to Many (e.g., image captioning).
  * Many to One (e.g., sentiment analysis of a tweet).
  * Many to Many (e.g., machine translation).

**Math in Simple RNNs:**

  * **Hidden State Update:** $h^{\<t\>} = g(W\_{h}[h^{\<t-1\>}, x^{\<t\>}]) + b\_h$ (where $g$ is an activation function like tanh). Can also be written as $h\_{\<t\>} = g(W\_{hh}h^{\<t-1\>} + W\_{hx}x^{\<t\>} + b\_h)$.
  * **Prediction:** $\\hat{y}^{\<t\>} = g(W\_{yh}h^{\<t\>} + b\_y)$.
  * Trainable parameters: $W\_{hh}$, $W\_{hx}$, $W\_{yh}$, $b\_{h}$, $b\_{y}$.

**Cost Function for RNNs:**

  * **Cross-Entropy Loss:** $J = -\\sum\_{j=1}^{K} y\_{j}^{\<t\>} \\log \\hat{y}\_{j}^{\<t\>}$ at each time step.
  * **Loss over Sequence:** $J = -\\frac{1}{T} \\sum\_{t=1}^{T} \\sum\_{j=1}^{K} y\_{j}^{\<t\>} \\log \\hat{y}\_{j}^{\<t\>}$ (average loss over all time steps).

**Gated Recurrent Units (GRUs):**

  * Similar to RNNs but have "update" ($\\Gamma\_u$) and "relevance" ($\\Gamma\_r$) gates to better control information flow and preserve important information over longer sequences.
  * Equations for GRU update:
      * $\\Gamma\_u = \\sigma(W\_u [h^{\<t-1\>}, x^{\<t\>}] + b\_u)$
      * $\\Gamma\_r = \\sigma(W\_r [h^{\<t-1\>}, x^{\<t\>}] + b\_r)$
      * $h^{\<t\>} = \\tanh(W\_h [\\Gamma\_r \* h^{\<t-1\>}, x^{\<t\>}] + b\_h)$
  * $\\sigma$ is the sigmoid activation function, tanh is the hyperbolic tangent activation function (output between -1 and 1, often used to introduce non-linearity and center the output).

**Deep and Bi-directional RNNs:**

  * **Bi-directional RNNs:** Process the input sequence in both forward and backward directions, allowing the model to consider context from both before and after a word for better understanding. Predictions ($\\hat{y}$) are based on combining hidden states from both directions.
  * **Deep RNNs:** Stack multiple RNN layers, allowing the model to learn more complex hierarchical features. The output of one layer becomes the input to the next. Each layer has its own weight matrices and activation functions.

**Implementation Note (Scan Function):**

  * The `scan` function (often found in libraries like Theano) abstractly represents the operation of an RNN, taking an initial state and a sequence, and iteratively applying a function to produce a sequence of outputs and a final state. This allows for efficient computation.


### Week 2 RNNs, Vanishing Gradients, LSTMs, and Named Entity Recognition:

**RNNs and Vanishing Gradients:**

* **Advantages of RNNs:** Capture short-range dependencies, less RAM than large N-gram models.
* **Disadvantages of RNNs:** Struggle with long-term dependencies, prone to vanishing or exploding gradients.
* **Vanishing Gradients:** During backpropagation through time, gradients can become very small due to repeated multiplication by values less than 1 (e.g., derivatives of sigmoid and tanh), hindering learning of long-range dependencies.
* **Solutions to Vanishing Gradients:** Use ReLU activation (derivative is 1 or 0), use Gated Recurrent Units (GRUs) or Long Short-Term Memory (LSTMs).

**Introduction to LSTMs:**

* LSTMs can "remember" and "forget" information, mitigating vanishing gradients.
* They have a cell state (long-term memory) and a hidden state (short-term memory), controlled by three gates:
    * **Input Gate:** Controls how much new information enters the cell state.
    * **Forget Gate:** Controls how much existing information is removed from the cell state.
    * **Output Gate:** Controls how much information from the cell state is output to the hidden state.
* Applications of LSTMs: Machine translation, speech recognition, text generation, sentiment analysis.
* **Key Idea:** The cell state acts as a conveyor belt, allowing information to flow through time with minimal linear interactions. Gates (sigmoid layer + pointwise multiplication) selectively add or remove information from the cell state.

**LSTM Architecture:**

* Consists of forget gate, input gate, output gate, cell state, and hidden state.
* **Forget Gate ($f$):** $\sigma(W_f[h_{t-1};x_t]+b_f)$ - decides what information to discard from the cell state.
* **Input Gate ($i$) and Gate Gate ($g$):** $i=\sigma(W_i[h_{t-1};x_t]+b_i)$, $g=tanh(W_g[h_{t-1};x_t]+b_g)$ - decide what new information to store in the cell state.
* **Cell State ($c_t$):** $f \odot c_{t-1} + i \odot g$ - updated cell state.
* **Output Gate ($o$):** $\sigma(W_o[h_{t-1};x_t]+b_o)$ - decides what information to output.
* **Hidden State ($h_t$):** $o_t \odot tanh(c_t)$ - the output of the LSTM unit.

**Introduction to Named Entity Recognition (NER):**

* NER identifies and extracts predefined entities (e.g., places, organizations, names, dates) from text.
* Applications: Search efficiency, recommendation engines, customer service, automatic trading.

**Training NERs: Data Processing:**

* Convert words and entity classes into numerical arrays.
* Pad sequences with `<PAD>` tokens to ensure uniform length.
* Create a data generator to feed data in batches.
* Assign a unique number to each word and each entity class.

**Training an NER system:**

1. Create tensors for input words and their corresponding entity labels.
2. Batch the data.
3. Feed the batches into an LSTM layer.
4. Pass the LSTM output through a dense layer to get predictions for each word's entity class.
5. Use a log softmax activation over the K entity classes to get probability distributions.
* The architecture can vary.

**Computing Accuracy for NER:**

1. Pass the test set through the trained NER model.
2. Get the predicted entity class for each word by taking the argmax of the prediction array.
3. Mask out the predictions for padded tokens to avoid including them in the evaluation.
4. Compare the predicted entity labels with the true labels to calculate accuracy.

### Week 3 Siamese Networks and One-Shot Learning

**Siamese Network:**

* **Concept:** Learns what makes two inputs the same, unlike classification which learns what makes an input what it is.
* **Example 1 (Same Meaning, Different Words):** Highlights the need to go beyond exact word matching.
* **Example 2 (Different Meaning, Similar Words):** Shows that word overlap isn't sufficient for semantic similarity.
* **Applications:** Face recognition, signature verification, question deduplication, paraphrase detection.

**Architecture:**

* Consists of two identical sub-networks (sister networks) that process the two input data points.
* Sub-networks share the same parameters (weights are tied), meaning only one set of weights is trained.
* The output of each sub-network is a vector representation of the input.
* A similarity score (e.g., cosine similarity) is calculated between the output vectors to determine the similarity of the inputs. LSTMs are not always a component.

**Cost Function (Triplet Loss):**

* Uses triplets of data: Anchor (A), Positive (P - similar to A), and Negative (N - dissimilar to A).
* **Goal:** Minimize the distance (maximize similarity) between the anchor and positive examples, and maximize the distance (minimize similarity) between the anchor and negative examples.
* Aims to satisfy: $-cos(A,P) + cos(A,N) \leq 0$.
* Visual representation helps understand the relationship between cosine similarity scores.

**Triplets and Margin:**

* **Margin ($\alpha$):** Added to the cost function to create a "safety" margin: $Cost = max(-cos(A,P) + cos(A,N) + \alpha, 0)$.
* **Triplet Types:**
    * **Easy Negative:** $cos(A,N) < cos(A,P)$.
    * **Semi-hard Negative:** $cos(A,N) < cos(A,P) < cos(A,N) + \alpha$.
    * **Hard Negative:** $cos(A,P) < cos(A,N)$.

**Computing the Cost:**

* Batches are prepared with similar examples paired together within the batch.
* A similarity matrix is calculated between the output vectors of the two input columns.
* The diagonal of the matrix represents similarities between positive pairs.
* Off-diagonals represent similarities between anchor and negative pairs.
* Two cost components are calculated:
    * $Cost1 = max(-cos(A,P) + mean\_neg + \alpha, 0)$ (using the mean of negative similarities).
    * $Cost2 = max(-cos(A,P) + closest\_neg + \alpha, 0)$ (using the highest negative similarity).
* The total cost is $Cost1 + Cost2$.

**One-Shot Learning:**

* Scenario where the model needs to verify similarity with only one or a few examples of a new class (e.g., signature verification).
* Instead of training a classifier with many classes, a Siamese network learns a similarity function.
* **Process:**
    1. Convert two inputs into numerical vectors.
    2. Feed them into the trained Siamese model.
    3. Calculate the cosine similarity between the output vectors.
    4. Compare the similarity score against a predefined threshold ($\tau$) to determine if the inputs are similar.

**Training / Testing:**

* **Training:** Prepare batches of paired input vectors and train the Siamese network using the triplet loss.
* **Testing:** Convert the two inputs to be compared into vectors, feed them through the trained Siamese network, and compare their output vectors using cosine similarity against a threshold.

## Course 4 NLP with attention models
Neural Machine Translation (NMT) using Seq2Seq with Attention and Transformers, including evaluation metrics (BLEU, ROUGE), decoding strategies (Greedy, Random, Beam Search, MBR), and an introduction to Transformer-based models like BERT and T5, along with their applications in Question Answering and Summarization.

**Neural Machine Translation (NMT):**

* **Seq2Seq Model:** Maps variable-length input sequences to fixed-length memory and then to variable-length output sequences using an Encoder-Decoder architecture (typically LSTMs or GRUs).
    * **Encoder:** Processes the input sequence into a fixed-length context vector.
    * **Decoder:** Generates the output sequence based on the context vector and previously generated words.
    * **Shortcoming (Information Bottleneck):** Fixed-length memory struggles with long sequences, leading to information loss.
    * **Workaround:** Using encoder hidden states for each word, but has memory and context flaws.
* **Seq2Seq with Attention:** Addresses the information bottleneck by allowing the decoder to focus on specific parts of the input sequence at each decoding step.
    * Calculates weights for each encoder hidden state based on the decoder's previous hidden state.
    * Uses these weights to create a context vector that is a weighted sum of encoder hidden states, focusing attention on relevant input words.
    * Involves comparing decoder states with encoder states to determine important inputs.
    * The attention layer uses queries (from decoder), keys (from encoder), and values (from encoder hidden states).

**Transformers:**

* **Advantages over RNNs:** Parallel computation (no sequential dependency), more stable gradient propagation (solves vanishing gradient), better handling of long-range dependencies.
* **Key Components:**
    * **Scaled Dot-Product Attention:** Efficiently computes attention using matrix multiplications and softmax, scaled by the square root of the key dimension.
    * **Multi-Head Attention:** Runs multiple parallel attention mechanisms (heads) with different linear transformations to capture various relationships between words. The outputs of the heads are concatenated and linearly projected.
    * **Encoder-Decoder Structure:**
        * **Encoder:** Uses Self-Attention to compute contextual representations of the input, followed by residual connections, normalization, and feed-forward networks, repeated N times.
        * **Decoder:** Similar structure but uses Masked Self-Attention (prevents attending to future tokens) and an attention mechanism over the encoder's output, repeated N times.
    * **Positional Encoding:** Adds information about the position of words in the sequence since Transformers don't inherently process sequential data.

**Machine Translation Setup:**

* Uses pre-trained word embeddings or initializes with one-hot vectors.
* Maintains index mappings (word2ind, ind2word).
* Uses `<EOS>` (End of Sequence) tokens.
* Pads sequences with zeros for consistent length.
* **Teacher Forcing:** During training, the decoder's input at each step is the true target word, not the model's prediction, to prevent error accumulation and speed up convergence.
* **Curriculum Learning (Scheduled Sampling):** Gradually transitions from using true labels (Teacher Forcing) to using the model's own predictions as input during training to improve robustness in inference.

**NMT Model with Attention (Detailed Steps):**

1.  **Data Preparation:** Copies input and target sequences.
2.  **Encoder:** Embeds input tokens, uses LSTM to get hidden states, generates Keys and Values for attention.
3.  **Pre-Attention Decoder:** Embeds target tokens (right-shifted with a start token), uses LSTM to generate Queries for attention (Teacher Forcing).
4.  **Attention Mechanism:** Uses Queries, Keys, and Values to compute Context Vectors, applying a padding mask.
5.  **Post-Attention Decoder:** Uses LSTM with Context Vectors for final decoding.
6.  **Output:** Uses a dense layer and LogSoftmax to get probability distribution for the translated output.

**Evaluation Metrics:**

* **BLEU (Bilingual Evaluation Understudy):** Measures n-gram precision of the candidate translation compared to one or more reference translations. Uses a brevity penalty for short translations. Focuses on how many candidate n-grams are present in the reference.
* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Measures n-gram recall, i.e., how many n-grams from the reference translation are present in the candidate translation. Focuses on how much of the reference is captured in the candidate.
* **F1 Score:** Can be calculated by combining BLEU (precision) and ROUGE (recall) to get a balanced evaluation.

**Sampling and Decoding Strategies:**

* **Greedy Decoding:** Chooses the highest probability word at each step. Simple and fast but can lead to repetitions and suboptimal overall sequences.
* **Random Sampling:** Randomly selects the next word based on the probability distribution. Generates more diverse output but can be too random and incoherent.
    * **Temperature Scaling:** Adjusts the randomness of sampling by scaling the logits before applying softmax.
* **Beam Search:** Keeps track of the top B most likely sequences (beams) at each step, exploring multiple possibilities. Better than greedy but computationally more expensive and can favor shorter sentences.
* **Minimum Bayes Risk (MBR) Decoding:** Generates multiple candidate translations (e.g., through random sampling) and selects the one that has the highest average similarity (e.g., based on ROUGE) with other candidates, aiming for a consensus translation. More robust but computationally intensive.

**Transformers Decoder (GPT-2):**

* Predicts the next word autoregressively.
* **Architecture:** Input Embedding (token + positional), N Decoder Blocks (Masked Multi-Head Attention + Residual + Layer Norm, Feed-Forward Network + Residual + Layer Norm), Final Dense + Softmax.
* **Input Processing:** Tokenization, Embedding, Positional Encoding, Shift Right (adds start token).
* **Decoder Block:** Masked Self-Attention (prevents attending to future), Residual Connections, Layer Normalization, Feed-Forward Network (with ReLU and dropout).
* **Output Layer:** Fully Connected layer to vocabulary size, Softmax for probability distribution.

**Transformer Summarizer:**

* **Input Format:** \[News Article] + EOS + \[Summary] + EOS.
* **Loss Function (Weighted Cross-Entropy):** Only calculates loss on the summary part of the input to focus training on summarization. Can have small weights for the article part in low-data scenarios.
* **Inference:** Input only the \[News Article] + EOS, and the model autoregressively generates the summary until an EOS token is produced. Uses sampling strategies during generation.

**Question and Answering (Transfer Learning with Transformers):**

* Transfer learning in NLP (reduce training time, improve predictions, use smaller datasets) can be feature-based (using pre-trained embeddings) or fine-tuning (using the entire pre-trained model on a new task).
* Models like ELMo, GPT, BERT, and T5 leverage transfer learning.
* **BERT (Bidirectional Encoder Representations from Transformers):**
    * Pre-trained on unlabeled data using Masked Language Modeling (predicting masked words) and Next Sentence Prediction.
    * Input embeddings are the sum of token, segmentation, and positional embeddings.
    * Fine-tuned on various downstream tasks by adding task-specific output layers.
* **T5 (Text-to-Text Transfer Transformer):**
    * Frames all NLP tasks as text-to-text, using a unified architecture.
    * Pre-trained using a masking strategy similar to BERT, but the loss is on the target sequence.
    * Uses a standard Transformer encoder-decoder architecture.
    * Variants include Language Model (causal masking), Prefix Language Model (visible input, causal target).
* Multi-task training (as in T5) allows a single model to perform various NLP tasks.