# CineMatch Movie Recommendation System --- Technical Report

## 1. Executive Summary

CineMatch is a movie recommendation system developed using the MovieLens
32M dataset. The project explores several recommendation approaches and
combines them into a hybrid recommendation framework.

Three primary recommendation signals were developed:

1.  Content-based similarity using movie genres and user-generated tags.
2.  Collaborative filtering using user-rating behavior and truncated
    SVD.
3.  Also-rated similarity using movie co-rating patterns.

A hybrid recommender combines these signals using weighted score
blending. The models were evaluated using a temporal leave-one-out
protocol with ranking metrics.

The evaluation showed that collaborative filtering achieved the
strongest performance among the tested models, while the hybrid
recommender performed closely behind.

The final system is exposed through a Streamlit application named
**CineMatch**.

------------------------------------------------------------------------

## 2. Project Objectives

The main objectives were to:

-   Understand the MovieLens 32M dataset.
-   Explore movie, rating, user, genre, and tag behavior.
-   Prepare the data for recommendation modeling.
-   Build a content-based recommendation system.
-   Build a personalized collaborative-filtering system.
-   Build an also-rated recommendation system.
-   Combine the models into a hybrid recommender.
-   Evaluate the models quantitatively.
-   Deploy the recommendation system through Streamlit.

------------------------------------------------------------------------

## 3. Dataset

The project uses the MovieLens 32M dataset.

### 3.1 Dataset Components

The main files are:

-   `movies.csv`
-   `ratings.csv`
-   `tags.csv`
-   `links.csv`

### 3.2 Dataset Size

The original dataset contains:

-   87,585 movies
-   200,948 users
-   32,000,204 ratings
-   2,000,072 tag records

The ratings cover dates from January 1995 to October 2023.

The overall average rating is approximately 3.54.

### 3.3 Movie Metadata

The movie metadata contains:

-   Movie ID
-   Movie title
-   Genres

The dataset contains 20 genre categories when the `(no genres listed)`
placeholder is included.

Drama is the most common genre after exploding multi-genre records.

### 3.4 User-Generated Tags

The tag dataset contains approximately 141,000 unique tags.

Common tags include:

-   sci-fi
-   atmospheric
-   action
-   comedy

Tags provide additional semantic information for the content-based
model.

------------------------------------------------------------------------

## 4. Exploratory Data Analysis

Several patterns were identified during exploratory analysis.

### 4.1 Rating Distribution

The rating scale ranges from 0.5 to 5.0 in increments of 0.5.

A rating of 4.0 is the most frequent individual rating.

Ratings between 4.0 and 5.0 account for approximately 49.8% of all
ratings, while ratings between 0.5 and 2.0 account for approximately
12.6%.

### 4.2 User Activity

The average user has approximately 159 ratings.

The median user has 73 ratings.

The most active user has more than 33,000 ratings.

Approximately 98.2% of users have 1,000 or fewer ratings.

This indicates a highly uneven distribution of user activity.

### 4.3 Movie Activity

The median movie has only 5 ratings, while some movies have more than
100,000 ratings.

Approximately 94.79% of movies have 1,000 or fewer ratings.

This long-tail structure is important for recommendation-system design.

### 4.4 Popularity and Rating Quality

The correlation between movie rating count and average rating was
approximately 0.098.

This is a very weak positive relationship, suggesting that popularity
alone does not strongly determine average rating.

------------------------------------------------------------------------

## 5. Data Preprocessing

The preprocessing stage prepared separate representations for
content-based and collaborative models.

### 5.1 Movie Metadata

Movie genres were cleaned by replacing:

``` text
(no genres listed)
```

with an empty value.

Release years were extracted from movie titles.

There were 615 movies without an identifiable release year.

### 5.2 Collaborative Filtering Data

An iterative activity threshold was applied:

-   Minimum 50 ratings per user.
-   Minimum 10 ratings per movie.

The resulting collaborative-filtering dataset contains:

-   29,542,223 ratings
-   128,255 users
-   31,701 movies

This filtering reduces extreme sparsity and focuses the collaborative
model on sufficiently active users and movies.

### 5.3 Content Representation

Genres and cleaned user-generated tags were combined into a content
field.

The resulting content dataset contains 87,585 movies.

Some movies have no available content features. These movies are
retained because they may still be useful for collaborative filtering.

------------------------------------------------------------------------

## 6. Content-Based Recommendation Model

The content-based model represents each movie using its genres and tags.

### 6.1 TF-IDF

TF-IDF was used to transform movie text into numerical vectors.

Configuration:

``` python
TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
```

The resulting matrix has:

-   87,585 movies
-   248,779 features
-   approximately 2.91 million non-zero values

The matrix is extremely sparse, so dense conversion and a full
movie-by-movie similarity matrix were avoided.

### 6.2 Cosine Similarity

Cosine similarity is used to compare movie vectors.

Movies with similar genres and tags receive higher similarity scores.

Example results for **Toy Story** included:

-   Toy Story 2
-   A Bug's Life
-   Toy Story 3
-   Monsters, Inc.
-   Finding Nemo

This demonstrates that the content model can identify semantically
related movies.

------------------------------------------------------------------------

## 7. Collaborative Filtering Model

Collaborative filtering uses user-rating behavior instead of movie
metadata.

### 7.1 User-Movie Matrix

A sparse user-movie rating matrix was created with:

-   128,255 users
-   31,701 movies
-   29,542,223 observed ratings

The matrix is highly sparse, with approximately 99.27% missing entries.

### 7.2 Truncated SVD

Truncated SVD with 50 latent factors was used to learn lower-dimensional
representations.

``` python
TruncatedSVD(
    n_components=50,
    random_state=42
)
```

The model captured approximately 34.8% of the variance in the evaluation
setting.

The learned user factors were saved separately so the Streamlit
application can generate personalized recommendations without loading
the full 29.5-million-row ratings dataset.

------------------------------------------------------------------------

## 8. Also-Rated Recommendation Model

A movie-user interaction matrix was constructed using binary
interactions.

A value of 1 represents that a user rated a movie.

For a selected movie, the system identifies users who rated it and
counts the other movies those users also rated.

A normalized co-rating similarity is then calculated.

This produces an "People Also Rated" style recommendation.

An important terminology limitation is that MovieLens does not contain
actual viewing logs. Therefore, this signal represents **co-rating
behavior**, not confirmed watch behavior.

------------------------------------------------------------------------

## 9. Hybrid Recommendation Model

The hybrid recommender combines three recommendation sources:

-   Content-based similarity
-   Collaborative filtering
-   Also-rated similarity

The initial configuration was:

``` text
Content weight              = 0.30
Collaborative weight        = 0.50
Also-rated weight            = 0.20
```

Scores from the different models are normalized before weighted
blending.

The candidate-generation process takes the top candidates from each
model, combines them, removes the selected movie and known user-rated
movies where available during evaluation, and then ranks candidates
using the hybrid score.

The weights were treated as initial configuration values rather than
optimized hyperparameters.

------------------------------------------------------------------------

## 10. Model Evaluation

### 10.1 Evaluation Strategy

A temporal leave-one-out evaluation protocol was used.

For each user:

1.  Ratings were sorted chronologically.
2.  The user's most recent rating was selected as the test interaction.
3.  Earlier ratings were retained as training data.
4.  Ratings of 4.0 or higher were considered positive.
5.  Top-10 recommendations were evaluated.

The positive test set contained 73,484 relevant interactions.

### 10.2 Evaluation Metrics

The following ranking metrics were used:

-   Hit Rate@10
-   Precision@10
-   Recall@10
-   NDCG@10

### 10.3 Results

  Model                       Hit Rate@10   Precision@10   Recall@10   NDCG@10
  ------------------------- ------------- -------------- ----------- ---------
  Content-Based                     3.50%          0.35%       3.50%     1.82%
  Collaborative Filtering          10.12%          1.01%      10.12%     5.27%
  Hybrid                            9.26%          0.93%       9.26%     4.98%

### 10.4 Interpretation

Collaborative filtering achieved the strongest performance across the
reported metrics.

The hybrid model performed slightly below collaborative filtering:

-   Hit Rate@10 decreased from 10.12% to 9.26%.
-   NDCG@10 decreased from 5.27% to 4.98%.

The content-based model had lower standalone performance under this
evaluation protocol.

However, the content model remains valuable because it provides a
complementary signal and can help with movies for which behavioral
information is limited.

The results also indicate that the initial hybrid weights should not be
considered optimal.

------------------------------------------------------------------------

## 11. Streamlit Application

The final recommendation system is exposed through a Streamlit
application called **CineMatch**.

The application provides:

### Movie Selection

Users can select a movie through a searchable dropdown.

### Movies You May Like

Provides content-based recommendations using genres and tags.

### People Also Rated

Provides recommendations based on co-rating behavior.

### Recommended For You

Provides personalized recommendations using the learned
collaborative-filtering representations.

The application uses saved model artifacts rather than loading the full
ratings dataset, which keeps the frontend substantially lighter.

------------------------------------------------------------------------

## 12. Model Artifacts

The project saves the following model and data artifacts:

``` text
models/
├── tfidf_matrix.npz
├── tfidf_vectorizer.joblib
├── collaborative_svd.joblib
├── user_to_index.joblib
├── movie_to_index.joblib
├── user_ids.npy
├── movie_ids.npy
├── user_factors.npy
├── movie_user_matrix.npz
├── movie_ids_aw.npy
├── movie_to_index_aw.joblib
├── movie_user_counts.csv
└── hybrid_config.joblib
```

------------------------------------------------------------------------

## 13. Limitations

### 13.1 Ratings Are Not Watch Logs

The MovieLens dataset records ratings, not actual viewing sessions.

Therefore, the also-rated model should not be interpreted as a true
"people also watched" model.

### 13.2 Hybrid Weights

The 30/50/20 weighting scheme was selected as an initial configuration.

It was not optimized through a validation-based search.

### 13.3 Cold Start

New users and movies remain challenging.

The content model can provide information for movies with metadata, but
a new user with no rating history remains difficult to personalize.

### 13.4 Personalized History in the Application

The Streamlit application uses saved user factors for efficient
personalization.

A lightweight precomputed user-history artifact could be added later to
explicitly exclude movies the selected user has already rated.

### 13.5 Evaluation Scope

The behavioral and hybrid evaluation involved sampled cases for
computational practicality. Therefore, those results should be
interpreted as evaluation estimates under the specified protocol rather
than universal production performance.

------------------------------------------------------------------------

## 14. Future Improvements

The following improvements would make the system more production-ready:

1.  Optimize hybrid weights using validation data.
2.  Add a lightweight user-rating-history lookup.
3.  Exclude already-rated movies in the Streamlit application.
4.  Add explainable recommendation reasons.
5.  Add movie posters and richer movie metadata.
6.  Experiment with implicit-feedback recommendation algorithms.
7.  Compare SVD with other matrix-factorization methods.
8.  Tune the number of latent factors.
9.  Evaluate additional ranking metrics.
10. Add online monitoring after deployment.
11. Deploy the Streamlit application.
12. Add automated model retraining.

------------------------------------------------------------------------

## 15. Conclusion

CineMatch demonstrates a complete recommendation-system workflow, from
raw data exploration and preprocessing to model development, evaluation,
and application deployment.

The project developed three complementary recommendation approaches and
combined them into a hybrid system.

The evaluation showed that collaborative filtering was the strongest
individual model under the current protocol, while the hybrid model
remained competitive.

The project also demonstrates practical data-science considerations
including sparse matrices, dimensionality reduction, candidate
generation, ranking metrics, computational constraints, and lightweight
deployment.

The next major technical improvements are hybrid-weight optimization,
user-history filtering, recommendation explanations, and richer movie
presentation.

------------------------------------------------------------------------

## 16. Technology Stack

-   Python
-   Pandas
-   NumPy
-   SciPy
-   Scikit-learn
-   Joblib
-   Matplotlib
-   Seaborn
-   Jupyter
-   Streamlit

## 17. Project Status

**Current status:** End-to-end recommendation system completed with
offline evaluation and Streamlit deployment.

The application frontend is being refined separately from the trained
recommendation models.
