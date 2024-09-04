# Choosing the Best Model for Resume Categorization

In this documentation, I will discuss the rationale behind selecting the BERT model as the best choice for the given resume dataset and the analysis of the various models tested.

## Dataset Overview:
The dataset provided for this project contains resumes categorized into different domains. The goal is to train a model that can accurately predict the category of each resume. The dataset poses several challenges, such as a relatively low amount of data and an imbalance among different categories.

## Model Selection and Rationale:

### 1. KNN

   -  KNN does not learn any model parameters, so it cannot generalize from the training data in the same way as BERT, making it less suited for complex NLP tasks.

   ```
                        precision    recall  f1-score   support

            ACCOUNTANT       0.63      0.93      0.75        29
              ADVOCATE       0.50      0.43      0.46        30
           AGRICULTURE       1.00      0.12      0.22         8
               APPAREL       0.46      0.30      0.36        20
                  ARTS       0.00      0.00      0.00        18
            AUTOMOBILE       0.33      0.17      0.22         6
              AVIATION       0.65      0.62      0.63        21
               BANKING       0.78      0.61      0.68        23
                   BPO       0.00      0.00      0.00         2
  BUSINESS-DEVELOPMENT       0.49      0.67      0.56        27
                  CHEF       0.77      0.71      0.74        24
          CONSTRUCTION       0.83      0.71      0.76        34
            CONSULTANT       0.33      0.20      0.25        20
              DESIGNER       0.79      0.79      0.79        19
         DIGITAL-MEDIA       0.75      0.48      0.59        25
           ENGINEERING       0.63      0.57      0.60        21
               FINANCE       0.38      0.32      0.34        19
               FITNESS       0.83      0.26      0.40        19
            HEALTHCARE       0.34      0.50      0.41        20
                    HR       0.53      0.89      0.67        18
INFORMATION-TECHNOLOGY       0.45      0.88      0.60        26
      PUBLIC-RELATIONS       0.45      0.59      0.51        17
                 SALES       0.44      0.55      0.49        29
               TEACHER       0.56      0.68      0.61        22

              accuracy                           0.56       497
             macro avg       0.54      0.50      0.49       497
          weighted avg       0.57      0.56      0.54       497


   ```

### 2. Decision Tree:
   - The structure of a Decision Tree is inherently linear and cannot effectively model the intricate relationships between words, phrases, and their context. Moreover, Decision Trees are prone to overfitting, especially in high-dimensional spaces, which can limit their effectiveness in NLP tasks compared to BERT, which excels at generalizing from large amounts of data and capturing deeper linguistic features.

   ```
                        precision    recall  f1-score   support

            ACCOUNTANT       0.89      0.86      0.88        29
              ADVOCATE       0.80      0.67      0.73        30
           AGRICULTURE       0.18      0.25      0.21         8
               APPAREL       0.29      0.25      0.27        20
                  ARTS       0.21      0.28      0.24        18
            AUTOMOBILE       0.00      0.00      0.00         6
              AVIATION       0.55      0.76      0.64        21
               BANKING       0.59      0.57      0.58        23
                   BPO       0.00      0.00      0.00         2
  BUSINESS-DEVELOPMENT       0.38      0.37      0.38        27
                  CHEF       0.83      0.62      0.71        24
          CONSTRUCTION       0.85      0.85      0.85        34
            CONSULTANT       0.50      0.35      0.41        20
              DESIGNER       0.65      0.79      0.71        19
         DIGITAL-MEDIA       0.62      0.40      0.49        25
           ENGINEERING       0.88      0.67      0.76        21
               FINANCE       0.60      0.47      0.53        19
               FITNESS       0.65      0.68      0.67        19
            HEALTHCARE       0.60      0.75      0.67        20
                    HR       0.67      0.67      0.67        18
INFORMATION-TECHNOLOGY       0.53      0.65      0.59        26
      PUBLIC-RELATIONS       0.47      0.53      0.50        17
                 SALES       0.44      0.41      0.43        29
               TEACHER       0.62      0.59      0.60        22

              accuracy                           0.58       497
             macro avg       0.53      0.52      0.52       497
          weighted avg       0.60      0.58      0.58       497



   ```

### 3. Random Forest:
   - The bagging technique used in Random Forests can help with variance reduction, but it does not address the need to understand the sequential nature and context of text. BERT, on the other hand, is specifically designed to handle sequential data and can leverage its transformer architecture to capture long-range dependencies and context, making it more effective for NLP tasks than Random Forest.

   ```

                        precision    recall  f1-score   support

            ACCOUNTANT       0.69      0.93      0.79        29
              ADVOCATE       0.87      0.67      0.75        30
           AGRICULTURE       0.67      0.25      0.36         8
               APPAREL       0.82      0.45      0.58        20
                  ARTS       0.20      0.06      0.09        18
            AUTOMOBILE       0.00      0.00      0.00         6
              AVIATION       0.82      0.86      0.84        21
               BANKING       0.77      0.74      0.76        23
                   BPO       0.00      0.00      0.00         2
  BUSINESS-DEVELOPMENT       0.80      0.59      0.68        27
                  CHEF       0.82      0.75      0.78        24
          CONSTRUCTION       0.87      0.79      0.83        34
            CONSULTANT       0.50      0.20      0.29        20
              DESIGNER       0.73      0.84      0.78        19
         DIGITAL-MEDIA       0.88      0.60      0.71        25
           ENGINEERING       0.52      0.71      0.60        21
               FINANCE       0.50      0.32      0.39        19
               FITNESS       0.69      0.58      0.63        19
            HEALTHCARE       0.41      0.60      0.49        20
                    HR       0.52      0.83      0.64        18
INFORMATION-TECHNOLOGY       0.51      0.96      0.67        26
      PUBLIC-RELATIONS       0.57      0.71      0.63        17
                 SALES       0.59      0.69      0.63        29
               TEACHER       0.55      0.82      0.65        22

              accuracy                           0.65       497
             macro avg       0.60      0.58      0.57       497
          weighted avg       0.66      0.65      0.63       497

   ```

### 4. Bagging:

   - Bagging (Bootstrap Aggregating) is an ensemble technique that improves the stability and accuracy of machine learning algorithms by combining the outputs of multiple models trained on different subsets of the data. While bagging can enhance the performance of models like Decision Trees and SVMs, it still does not address the fundamental challenges of handling text data, such as understanding context and semantic meaning.

```
                        precision    recall  f1-score   support

            ACCOUNTANT       0.65      0.90      0.75        29
              ADVOCATE       0.47      0.27      0.34        30
           AGRICULTURE       0.00      0.00      0.00         8
               APPAREL       0.50      0.20      0.29        20
                  ARTS       0.17      0.11      0.13        18
            AUTOMOBILE       0.50      0.17      0.25         6
              AVIATION       0.80      0.57      0.67        21
               BANKING       0.70      0.61      0.65        23
                   BPO       0.00      0.00      0.00         2
  BUSINESS-DEVELOPMENT       0.18      0.81      0.29        27
                  CHEF       0.78      0.75      0.77        24
          CONSTRUCTION       0.79      0.56      0.66        34
            CONSULTANT       0.00      0.00      0.00        20
              DESIGNER       0.67      0.95      0.78        19
         DIGITAL-MEDIA       0.72      0.52      0.60        25
           ENGINEERING       0.82      0.43      0.56        21
               FINANCE       0.60      0.32      0.41        19
               FITNESS       1.00      0.26      0.42        19
            HEALTHCARE       0.44      0.40      0.42        20
                    HR       0.68      0.72      0.70        18
INFORMATION-TECHNOLOGY       0.50      0.77      0.61        26
      PUBLIC-RELATIONS       0.70      0.41      0.52        17
                 SALES       0.46      0.41      0.44        29
               TEACHER       0.64      0.64      0.64        22

              accuracy                           0.51       497
             macro avg       0.53      0.45      0.45       497
          weighted avg       0.57      0.51      0.50       497

```


   ### BERT
   - Achieve 79.0% accuracy
   - BERT is a state-of-the-art deep learning model designed specifically for natural language processing (NLP) tasks. Unlike traditional machine learning models, BERT uses a transformer-based architecture that allows it to capture complex linguistic patterns, semantics, and context in text data. One of the key advantages of BERT over traditional models is its ability to understand the context of a word in a sentence by considering both its left and right surroundings simultaneously.
   ```

                           precision    recall  f1-score   support

            ACCOUNTANT       1.00      1.00      1.00        24
              ADVOCATE       0.95      0.79      0.86        24
           AGRICULTURE       1.00      0.38      0.56        13
               APPAREL       0.35      0.37      0.36        19
                  ARTS       0.53      0.43      0.47        21
            AUTOMOBILE       0.00      0.00      0.00         7
              AVIATION       0.55      0.75      0.63        24
               BANKING       0.67      0.61      0.64        23
                   BPO       0.00      0.00      0.00         4
  BUSINESS-DEVELOPMENT       0.82      0.96      0.88        24
                  CHEF       0.91      0.83      0.87        24
          CONSTRUCTION       0.95      0.82      0.88        22
            CONSULTANT       1.00      0.96      0.98        23
              DESIGNER       0.88      1.00      0.93        21
         DIGITAL-MEDIA       0.82      0.74      0.78        19
           ENGINEERING       0.96      0.96      0.96        24
               FINANCE       0.89      1.00      0.94        24
               FITNESS       0.71      0.52      0.60        23
            HEALTHCARE       0.65      0.74      0.69        23
                    HR       0.92      1.00      0.96        22
INFORMATION-TECHNOLOGY       0.60      1.00      0.75        24
      PUBLIC-RELATIONS       0.70      0.64      0.67        22
                 SALES       0.88      0.96      0.92        23
               TEACHER       0.86      0.95      0.90        20

              accuracy                           0.79       497
             macro avg       0.73      0.72      0.72       497
          weighted avg       0.78      0.79      0.77       497
```