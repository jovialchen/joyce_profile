
# Natural Language Processing Specialization


[Certificate](https://www.coursera.org/account/accomplishments/specialization/certificate/ECZE48SJP9B6)


## Course 1 NLP with Classification and Vector Spaces
### Week 1
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
