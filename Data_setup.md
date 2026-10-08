# Data Setup

This project uses the **MovieLens 32M dataset** to build a Netflix-style movie recommendation system.

The system supports:

* **Content-Based Recommendation** — finding movies similar to a selected movie.
* **Collaborative Filtering** — recommending movies based on user rating behavior.
* Movie metadata and user-rating analysis.

---

## 1. Dataset

The project uses the **MovieLens 32M dataset** provided by GroupLens Research.

Dataset name:

**MovieLens 32M**

The dataset contains approximately:

* **32 million ratings**
* **200,948 users**
* **87,585 movies**
* **2 million tags**

The dataset is relatively large, so it should be downloaded separately rather than stored in this GitHub repository.

---

## 2. Download the Dataset

Download the **MovieLens 32M** dataset from the official GroupLens website.

After downloading the ZIP file, extract it.

The extracted folder should contain the following files:

```text
ml-32m/
├── ratings.csv
├── movies.csv
├── tags.csv
├── links.csv
├── README.txt
└── checksums.md5
```

---

## 3. Project Data Structure

Place the extracted dataset inside the project's `raw` directory.

The expected structure is:

```text
Netflix Movie Recommendation System/data/
│
├── raw/
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   ├── links.csv
│   ├── README.txt
│   └── checksums.txt
│
├── notebooks/
├── src/
├── models/
├
├── app/
├── requirements.txt
├── README.md
├── DATA_SETUP.md
├── report.md
└── .gitignore
```

---

## 4. Dataset Files

### `movies.csv`

Contains movie metadata.

Columns:

| Column    | Description                       |
| --------- | --------------------------------- |
| `movieId` | Unique MovieLens movie identifier |
| `title`   | Movie title                       |
| `genres`  | Movie genres separated by `\|`    |

Example:

```text
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
```

This file is primarily used for the **content-based recommendation system**.

---

### `ratings.csv`

Contains user ratings for movies.

Columns:

| Column      | Description                                |
| ----------- | ------------------------------------------ |
| `userId`    | Unique user identifier                     |
| `movieId`   | Movie identifier                           |
| `rating`    | Rating from 0.5 to 5.0                     |
| `timestamp` | Unix timestamp when the rating was created |

Example:

```text
userId,movieId,rating,timestamp
1,296,5.0,1147880044
```

This is the main dataset used for **collaborative filtering and user preference analysis**.

Because this file is very large, it should remain outside GitHub.

---

### `tags.csv`

Contains user-generated tags associated with movies.

Columns:

| Column      | Description                    |
| ----------- | ------------------------------ |
| `userId`    | User who created the tag       |
| `movieId`   | Movie associated with the tag  |
| `tag`       | User-generated description/tag |
| `timestamp` | Unix timestamp                 |

Tags can provide additional information about movie characteristics and may be useful for future improvements to the content-based recommendation system.

---

### `links.csv`

Contains identifiers connecting MovieLens movies to external movie databases.

Columns:

| Column    | Description                |
| --------- | -------------------------- |
| `movieId` | MovieLens movie identifier |
| `imdbId`  | IMDb identifier            |
| `tmdbId`  | TMDB identifier            |

These identifiers can be useful if external movie information is added in a future version of the project.

---

### `README.txt`

The original documentation supplied with the MovieLens dataset.

It contains information about the dataset, licensing, files, and other details provided by GroupLens.

---

## 5. Dataset Validation

After placing the files in the `raw/` directory, verify that the following files exist:

```text
raw/
├── ratings.csv
├── movies.csv
├── tags.csv
├── links.csv
├── README.txt
└── checksums.txt
```

The expected approximate dataset sizes used during development were:

| File          | Approx. Size |
| ------------- | -----------: |
| `ratings.csv` |       836 MB |
| `tags.csv`    |        69 MB |
| `movies.csv`  |         4 MB |
| `links.csv`   |         2 MB |

Small differences in file size are normal.

---

## 6. Dataset Checks Performed

The dataset was explored and validated before building the recommendation models.

### Movies

* 87,585 movie records
* 3 columns
* No missing values in the main movie metadata
* 20 genre categories
* Movies range from 1874 to 2023 based on available release years
* 615 movies do not have a detectable release year from the title

### Ratings

* 32,000,204 rating records
* 200,948 unique users
* 84,432 unique rated movies
* Rating scale: 0.5–5.0
* No missing values
* Movie IDs in the ratings data were matched against the movie metadata

### Tags

* 2,000,072 tag records
* 15,848 users contributed tags
* 51,323 movies have tags
* 140,979 unique tag values

These checks help ensure that the dataset is suitable for the recommendation pipeline.

---

## 7. Why the Dataset Is Not Included in GitHub

The raw MovieLens dataset is intentionally excluded from this repository because it is large and is distributed separately by GroupLens.

In particular:

```text
ratings.csv
```

is approximately **836 MB**, which is too large for a normal GitHub repository.

The repository should therefore contain the **code and documentation required to reproduce the project**, while users download the dataset separately.

The `.gitignore` file should exclude the raw dataset files.

Example:

```gitignore
raw/*.csv
```

---

## 8. Reproducing the Data Setup

To reproduce the project:

1. Clone the repository.
2. Download the MovieLens 32M dataset.
3. Extract the dataset.
4. Create the `raw` directory if it does not already exist.
5. Place the dataset files inside `raw/`.
6. Install the required Python packages:

```bash
pip install -r requirements.txt
```

7. Open the notebooks and run
The project should then be able to access the data using paths such as:

```python
RAW_DATA_PATH = "raw"

movies_path = f"{RAW_DATA_PATH}/movies.csv"
ratings_path = f"{RAW_DATA_PATH}/ratings.csv"
tags_path = f"{RAW_DATA_PATH}/tags.csv"
links_path = f"{RAW_DATA_PATH}/links.csv"
```

---

## 9. Data Privacy and Licensing

MovieLens is a research dataset provided by **GroupLens Research**.

Users should download the dataset directly from the official source and follow the dataset's accompanying terms and conditions.

This repository does not redistribute the raw MovieLens dataset.

---

## 10. Quick Checklist

Before running the project, make sure:

* [ ] MovieLens 32M has been downloaded
* [ ] The dataset has been extracted
* [ ] `raw/` exists
* [ ] `movies.csv` is present
* [ ] `ratings.csv` is present
* [ ] `tags.csv` is present
* [ ] `links.csv` is present
* [ ] Raw CSV files are excluded from Git
* [ ] Required Python packages are installed
* [ ] The project notebooks/scripts can access the `raw/` directory

Once these steps are complete, the project is ready for data preprocessing and recommendation-model development.
