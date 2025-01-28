# Analyzing Student Stress Factors through Machine Learning Techniques

Authors: Zachary Perry and Manan Patel

## Dataset

[Student Stress Factors - A Comprehensive Analysis](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis)

1. Anxiety : range from 0 to 21, Measure : GAD-7
2. Self-esteem : range 0 to 30, Measure: Rosenberg Self Esteem Scale
3. Mental Health History : 0 if no mental health history, 1 if mental health history
4. Depression : range 0 to 27, Measure: Patient Health Questionnaire (PHQ-9)
5. Other features mostly range from 0 to 5 considering 0,1 to be low, 2,3 to be mid, and 4,5 to be high.
   However, these other features could also be given scientific measures. eg. blood pressure.

`stress_level` is a categorical variable with 3 levels: `low`, `medium`, `high`
Age Groups: 15-24 (High school through college), 1,100 students
City: Dharan, Nepal

## Research Questions

1. Can we predict the stress level of a new student based on the similarity of their stress factors to those of known students?
    - K-Nearest Neighbors (KNN)
      - K Fold Cross Validation (`n_neighbors`)
    - Multi-Layer Perceptron (MLP)
      - K-fold cross validation (`hidden_layer_sizes`, `learning_rate_init`)
2. Which factors contribute most significantly to predicting stress levels in students?
    - Decision Trees
      - Depth of Tree (`max_depth`)
    - Random Forest
      - Number of trees (`n_estimators`)
