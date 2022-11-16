# Paul The Octopus Machine Learning

Desing by Frank Laércio, Software Developer at CI&T.<br>
frank.junior@ciandt.com [github.com/franklaercio](https://github.com/franklaercio)

![banner.png](img/paul.png)

**Content**
1. Importing libraries and files for GCP
2. Fifa World Cup Data Analyzing
3. Historical Results
4. Ranking Fifa
5. Modeling Machine Learning for Fifa World Cup Predictions
6. Qatar World Cup Predictions

## **Fifa World Cup Data Analyzing**

In this section, we understand our dataset and make some assumptions. Like a does the FIFA ranking influence the results, are the clashes in world cups important in today's results or the win set is an interesting data.

### **Historical Results**

This section is an important part of building the algorithm, as we will check the previous results of the selections in matches. But we will make the following refinements to the data:
* Check match dates.
* Select the most played tournaments, as they are the most disputed.
* Remove friendlies, as they can negatively influence predictions, due to their competitiveness.
* Remove matches older than the 21st century, as soccer is very current.

### **Ranking Fifa**

In this section we will check the Fifa ranking for each team. This step will be important for determining the algorithm, as the FIFA ranking makes a calculation to rank the strongest teams.

## **Modeling Machine Learning for Fifa World Cup Predictions**

In this section we will make the predictions after organizing the data previously done. For this, we will use the decision tree technique and a regression forest. These are algorithms provided by the sklearn library. <br>
Para fazer o aprendizado iremos utilizar os dados da diferença de ranks e a média do rank. In addition, to verify the effectiveness of the algorithm, we will use the data of who won or lost the match.
