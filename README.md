# Machine Learning - 2017.2 Classes @ PUC-Rio

Code for machine learning classes at PUC-Rio

## Two Problems

- Given a product, give another product that was co-purchaseds

  { f (ASIN) -> {y : ASIN} | y is co-purchased } 

- Given a product pair, say if they were co-purchased

  f (ASIN, ASIN2) -> {yes, no}

## Methods to try:

- Dumb algorithm: say that a pair is always non co-purchased (for problem #2)
- Algorithms used by CSE190 classes from U.C. San Diego
  - Logistic Regression
  - Decision Tree
  - Adaptative Boost (AdaBoost)
  - K-Neighbors Classifier
  - Random Forest
- Neural Networks
- Support Vector Machine (SVM)

## Algorithms

### Decision Tree

|ASIN|title|group|salesrank|avg-rating-comments|

f(ASIN, title, ASIN2, title2) -> {yes, no}