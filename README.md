# IBM Article Recommendation System

## Overview
This project builds and evaluates multiple recommendation approaches using the **IBM user–article interactions** dataset. The goal is to recommend relevant articles to users based on **popularity**, **user behavior**, and **content similarity**, while handling common recommender challenges like **sparsity** and **cold-start**.

---

## Dataset
The dataset contains user–article interaction logs with (at least) the following fields:
- `user_id` (mapped from email)
- `article_id`
- `title`

**Important:** Each row is treated as a **single user–article interaction**. If a user interacts with the same article multiple times, each interaction is counted.

---

## Methods Implemented

### 1) Rank-Based Recommendations (Section 2)
**What it does:** Recommends the most popular articles overall (highest interaction count).  
**Strengths:** Simple, robust, and works well for **new users** with no history.  
**Limitations:** Not personalized and can over-recommend the same popular content (popularity bias).

---

### 2) User–User Collaborative Filtering (Section 3)
**What it does:** Finds similar users based on interaction overlap (e.g., cosine similarity) and recommends articles read by similar users that the target user hasn’t seen.  
**Strengths:** Personalized recommendations for **users with enough history**.  
**Limitations:** Struggles with **cold-start users**, can be sensitive to sparse data, and may be computationally heavier at scale.

---

### 3) Content-Based Recommendations (Section 4)
**What it does:** Builds article similarity from text features (titles). Typical pipeline:
- TF-IDF on titles  
- Dimensionality reduction (SVD/LSA)  
- Clustering (e.g., KMeans)  
Then recommends articles from the same cluster, often ranked by popularity.

**Strengths:** Works well for **new or lightly active users** and supports “more like this” recommendations.  
**Limitations:** Quality depends on text richness (titles alone can be limited) and can reduce novelty by recommending near-duplicates.

---

## Evaluation & Model Selection
TruncatedSVD was tested with varying latent dimensions and evaluated with **accuracy**, **precision**, and **recall** on reconstructing the user–item matrix.

A practical choice was around **200 latent features**, since:
- recall is already high (e.g., > 0.8),
- accuracy and precision are near 1,
- increasing dimensions further gives diminishing returns while making the model slower and more complex.

---

## Cold-Start Strategy (Practical Deployment)
A hybrid approach is recommended depending on user history:
- **No history:** Rank-based (popularity)
- **Little history:** Content-based (similar titles/topics) + popularity fallback
- **Lots of history:** Collaborative filtering (personalization) + content-based for diversity

---

## Improvements & Next Steps
Potential improvements to increase recommendation quality:
- Use richer text data: article summaries, full text, tags, keywords, author, categories.
- Use modern embeddings (sentence/document embeddings) instead of TF-IDF only.
- Build a hybrid ranker combining popularity + CF + content signals.
- Use online evaluation (A/B testing) with metrics such as CTR, completion rate, and retention.

---

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn (TF-IDF, TruncatedSVD, KMeans, cosine similarity)
- Jupyter Notebook
