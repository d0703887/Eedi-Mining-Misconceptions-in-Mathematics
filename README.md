# Intro
This repo contains our work for the Kaggle competition: Eedi-Mining Misconceptions in Mathematics. 

In this competition, we develop an NLP model to predict the affinity between misconceptions and incorrect answers in multiple-choice questions. 

For example:

![inbox_59561_c5f9c8305d36aa52a4ee8aad08ac7708_question_example](https://github.com/user-attachments/assets/8602fb6a-f2a5-4b36-aad7-9a22bb551220) 

If a student selects the distractor "13", they may have the misconception "*Carries out operations from left to right regradless of priority order."*

# Method
This competition posed two main challenges:
1. Complexity of Mathematical Content - The questions involve advanced mathematical reasoning.
2. Data Scacity and Imbalance - The dataset includes 2,500 types of misconceptions, but 700 of them do not appear in the training data. Additionally, one-third of the misconceptions appear only once in the training set.

To address these challenges:
- We utilized Qwen2.5, a LLM known for its strong mathematical reasoning abilities.
- To mitigate the data scarcity, we leveraged the LLM's reasoning capabilities to retireve possible misconceptions and used it to re-rank them, improving prediction accuracy. More details can be found in our report.

# Result
- Public Leaderboard Rank: 272nd / 1449 teams
- Private Leaderboard Rank: 609nd / 1449 teams
We observed that our model overfitted to the training dataset, leading to a drop in performance on the private leaderboard.
