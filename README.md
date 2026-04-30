# IBM Article Recommendation System

## Overview
This project builds and evaluates multiple recommendation approaches using the **IBM user–article interactions** dataset. The goal is to recommend relevant articles to users based on **popularity**, **user behavior**, and **content similarity**, while addressing common recommender challenges such as **sparsity** and **cold‑start users**.

---

## Dataset
The dataset contains user–article interaction logs with (at least) the following fields:
- `user_id` (mapped from email)
- `article_id`
- `title`

Each row is treated as a **single user–article interaction**. Repeated interactions between the same user and article are counted multiple times.

---

## Methods Implemented

### 1) Rank‑Based Recommendations (Section 2)
**Description:**  
Recommends the most popular articles overall based on total interaction counts.

**Characteristics:**
- No personalization
- Simple and robust
- Effective for users with no interaction history

---

### 2) User–User Collaborative Filtering (Section 3)
**Description:**  
Identifies users with similar interaction patterns (using similarity measures such as cosine similarity) and recommends articles read by similar users that the target user has not yet seen.

**Characteristics:**
- Personalized recommendations
- Requires sufficient user history
- Performance improves as interaction data increases

---

### 3) Content‑Based Recommendations (Section 4)
**Description:**  
Uses article title text to compute similarity between articles. Titles are vectorized using TF‑IDF, reduced with SVD (LSA), and clustered. Recommendations are generated from articles within the same cluster and ranked by popularity.

**Characteristics:**
- Works with little user history
- Supports topic‑based “similar article” recommendations
- Depends on the quality of textual features

---

### 5) Matrix Factorization (SVD‑Based Recommendations)
**Description:**  
Matrix factorization is applied to the user‑item interaction matrix using **TruncatedSVD**. This decomposes the interaction matrix into latent user and article factors that capture underlying interaction patterns.

Articles (or users) are then compared in this latent space using cosine similarity, enabling:
- Discovery of implicit relationships between articles
- Recommendations based on shared user behavior rather than explicit content or direct overlap

**Characteristics:**
- Captures latent structure in user behavior
- Reduces dimensionality and sparsity
- Provides a balance between scalability and expressiveness when choosing the number of latent features

---

## Model Evaluation
TruncatedSVD was evaluated by varying the number of latent features and measuring **accuracy**, **precision**, and **recall** when reconstructing the user–item interaction matrix.

A value of approximately **200 latent features** provided a good trade‑off:
- Recall exceeded 0.8
- Accuracy and precision were close to 1
- Increasing dimensionality further yielded diminishing returns while increasing computational cost

---

## Tech Stack
- Python
- Pandas, NumPy
- scikit‑learn (TF‑IDF, TruncatedSVD, KMeans, cosine similarity)
- Jupyter Notebook
