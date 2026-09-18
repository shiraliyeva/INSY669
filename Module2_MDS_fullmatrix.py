import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from matplotlib.colors import LinearSegmentedColormap


# Read the word-review matrix from CSV.

word_review_matrix = pd.read_csv("word_review_matrix.csv")

print("Word-review matrix:")
print(word_review_matrix.head())

print("\nShape:")
print(word_review_matrix.shape)


# Define our sentiment and category words.
# These are the words we will use to analyze the reviews.

negative_words = [
    "small",
    "bad",
    "disappoint",
    "rude",
    "expensive",
    "smell",
    "empty",
    "dirty",
    "rotten",
    "crowded",
    "avoid",
    "unpleasant",
    "discrimination",
    "spoiled",
    "overpriced",
    "horrible",
    "unacceptable",
    "poor",
    "expired",
    "broken",
    "waste",
    "messy",
    "inexperienced",
    "racist"
]

positive_words = [
    "good",
    "clean",
    "great",
    "well",
    "excellent",
    "organized",
    "helpful",
    "friendly",
    "welcoming",
    "perfect",
    "excited",
    "nice",
    "awesome",
    "love",
    "happy",
    "amazing"
]

sentiment_words = negative_words + positive_words


category_words = [
    "store",
    "maxi",
    "staff",
    "product",
    "price",
    "location",
    "service",
    "selection",
    "park",
    "cashier",
    "checkout",
    "shelves"
]


# Make sure we keep only the words that are actually present
# in the word-review matrix.

sentiment_words = [
    word for word in sentiment_words
    if word in word_review_matrix.columns
]

category_words = [
    word for word in category_words
    if word in word_review_matrix.columns
]


print("\nSentiment words found:")
print(sentiment_words)

print("\nCategory words found:")
print(category_words)


# Calculate our word counts and number of reviews.
# This will be used to calculate lift.

X = word_review_matrix

n_reviews = X.shape[0]

word_counts = X.sum(axis=0)

print("\nNumber of reviews:", n_reviews)


# Calculate co-occurrences.

# X.T @ X calculates how many reviews contain each pair of words.

cooccurrence = X.T @ X

cooccurrence_df = pd.DataFrame(
    cooccurrence,
    index=X.columns,
    columns=X.columns
)

print("\nExample co-occurrences:")
print(cooccurrence_df.loc[category_words, sentiment_words])


# Calculate lift for each pair of words.

# P(A)

p_word = word_counts / n_reviews


# P(A and B)

p_both = cooccurrence / n_reviews


# P(A) * P(B)

expected = np.outer(
    p_word.values,
    p_word.values
)


# Lift = P(A and B) / [P(A)P(B)]

lift = p_both / expected

lift_df = pd.DataFrame(
    lift,
    index=X.columns,
    columns=X.columns
)


print("\nCategory × Sentiment Lift:")
print(
    lift_df.loc[
        category_words,
        sentiment_words
    ]
)


# Keep only lift values for our category and sentiment words.
# This will be used to create sentiment-association profiles.

category_sentiment_lift = lift_df.loc[
    category_words,
    sentiment_words
]

print("\nCategory-Sentiment Lift Matrix:")
print(category_sentiment_lift)


# Each category word is now represented by its lift
# across all sentiment words.

sentiment_profiles = category_sentiment_lift.copy()

print("\nSentiment profiles:")
print(sentiment_profiles)


# Create a dissimilarity matrix for the category words
# based on their sentiment-association profiles.

# MDS needs distances between the CATEGORY WORDS.

# We calculate Euclidean distance between their
# sentiment-association profiles.

dissimilarity = pairwise_distances(
    sentiment_profiles,
    metric="euclidean"
)

dissimilarity_df = pd.DataFrame(
    dissimilarity,
    index=category_words,
    columns=category_words
)

print("\nDissimilarity Matrix:")
print(dissimilarity_df)


# Run MDS.

mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42
)

coordinates = mds.fit_transform(
    dissimilarity_df
)


# Create a DataFrame to hold the MDS coordinates
# for each category word.

mds_results = pd.DataFrame(
    coordinates,
    columns=["MDS1", "MDS2"],
    index=category_words
)

print("\nMDS coordinates:")
print(mds_results)


# Calculate an overall sentiment score for each category word based on its lift across the sentiment words.

# Total positive lift for each category.

positive_lift = category_sentiment_lift[
    positive_words
].sum(axis=1)


# Total negative lift for each category.

negative_lift = category_sentiment_lift[
    negative_words
].sum(axis=1)


# Sentiment score:
#
#       Positive Lift - Negative Lift
#       ------------------------------
#       Positive Lift + Negative Lift
#
# The result ranges from -1 to +1.
#
# -1 = completely negative
#  0 = balanced
# +1 = completely positive

sentiment_score = (
    positive_lift - negative_lift
) / (
    positive_lift + negative_lift
)


print("\nPositive Lift:")
print(positive_lift)

print("\nNegative Lift:")
print(negative_lift)

print("\nSentiment Scores:")
print(sentiment_score)


# Create a color map for the sentiment scores.

sentiment_cmap = LinearSegmentedColormap.from_list(
    "sentiment",
    [
        "red",
        "lightgray",
        "green"
    ]
)


# Convert the sentiment scores from [-1, +1]
# to the color scale range [0, 1].

color_values = (
    sentiment_score + 1
) / 2


# Plot MDS results

plt.figure(figsize=(12, 9))

plt.scatter(
    mds_results["MDS1"],
    mds_results["MDS2"],
    c=color_values,
    cmap=sentiment_cmap,
    vmin=0,
    vmax=1,
    s=150,
    edgecolors="black",
    linewidths=0.8
)


# Add category labels.

for word in mds_results.index:

    plt.annotate(
        word,
        (
            mds_results.loc[word, "MDS1"],
            mds_results.loc[word, "MDS2"]
        ),
        xytext=(7, 7),
        textcoords="offset points"
    )


# Add a colorbar to explain the sentiment scale.

cbar = plt.colorbar()

cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])

cbar.set_ticklabels([
    "Strong Negative",
    "Negative",
    "Balanced",
    "Positive",
    "Strong Positive"
])

cbar.set_label("Sentiment Association")


plt.xlabel("MDS Dimension 1")

plt.ylabel("MDS Dimension 2")

plt.title(
    "MDS of Category Words Based on Sentiment Associations"
)

plt.show()