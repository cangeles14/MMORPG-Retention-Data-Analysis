# Video Game Data Analysis with Life Beyond @ Ironhack Paris

Darewise is a company looking to create long-term and meaningful social experiences for players, changing how the world views online games

They are the developers behind Life Beyond, a currently in development, open world and persistent game where teamwork and player choice unlock rich social adventures.

## Purpose

Developing a successful video game may boil down to a mixture of the perfect gameplay with the perfect graphics...but what makes a game successful is tracking and analyzing the video game’s metrics.

Darewise is looking to explore their player base and bring insights to player retention.

The purpose of this project is to explore player actions and build two models. Using this information, I will present a strategy to increase player retention.

## Getting Started - Dataset

Database
- [BigQuery](https://cloud.google.com/bigquery/docs)
- [Python](https://docs.python.org/3/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)

Darewise was kind enough to allow me to work with thier data. To obtain the data and create a dataframe, I used Google Cloud Storage and BigQuery API to collect data from the players in Life Beyond.

From here, I explored what types of data I could retrieve and collected data on player actions within the game and used this data in Python and Pandas to create a dataframe.

Identifying what your players are doing in game is just as important as knowing your community outside your game.

Data was not only collected through players in game actions but also gamers that expressed interest in Life Beyond. 

## Data Exploration

Exploratory analysis is essential to diving deeper into the meaning of data trends, and extracting useful information from your dataset.

Here I used matplotlib and Tableau to take a first look at my dataframe. I wanted to see what is my data telling me and ask questions that my data can answer.

- DAU, MAU, & STICKINESS - How many players do we have a daily or monthly and which daily gamers are also monthly gamers?

- GAMEPLAY PREFERENCE - As an MMORPG, what types of gamers are playing Life Beyond?

- GAME ACTIVITY - What type of gamer is doing which activity in Life Beyond?

## Gameplay Overview and Visualizations
 
DAU or daily active users defines unique users who played the game within a single day. This can include players that played for hours or those that simply just logged in.

MAU or monthly active users are players that played at least once within a months time. Again a player simply has to log in as it doesn't matter the duration of play.

Stickiness rate is the ratio of monthly users that are also daily users. This can be seen as DAU/MAU. This metric is helpful to see the rate of players that are continuing to play overtime.

The session time or how long a player is playing in a sitting can be extremely helpful when discovering what your players are doing and what keeps them in the game.
 
## Prediction Model

Part of this project was to propose a prediction model to influence Darewise to increase player rentention.

To do this I created a Categorical Boost prediction model. This model uses the in game player activity and the players demographics to predict the average session playtime. By predicting the average session playtime Darewise will be able to invite new players that are more likely to have longer play sessions. This will both increase retention and also allow them to collect more data to increase their analysis on their playerbase.

Optimization is key to producing an effective and accurate model. To do this I tested multiple models and compared different metrics including recall, accuracy, precision, and F1 scores. Additionally, I compared the prediction statistics of each model via a confusion matrix.

## Conclusions

PLAYER STATISTICS - Since Life Beyond is still in development, only a select number of players are in game. This greatly effects the bias of data collected. To remove this bias, it is crucial to collect more data from more unique players.

NEW PLAYERS - With this in mind, numerous players have shown interest in playing. This is most benefitcial for Darewise as having a large interest in your game allows for potential to collect more data.

INVITE BEST PLAYERS - Using this prediction model, we can invite new players that will play Life Beyond the longest - increasing rentention rate by having longer playing players, but also allowing to collect more game data statistics.


## Built With

* [Python](https://docs.python.org/3/) - The programming language used
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language
* [Tableau](https://www.tableau.com/) - Popular Data visualization tool
* [MatPlotLib](https://matplotlib.org/contents.html) - Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms
* [BigQuery API](https://cloud.google.com/bigquery/docs)
* [CatBoost](https://catboost.ai/)
* [Scikit-Learn](https://scikit-learn.org/stable/)


## Authors

* **Christopher Angeles** - [cangeles14](https://github.com/cangeles14)

## Acknowledgments

* [Ironhack](https://www.ironhack.com/en/data-analytics) -  Data Analytics Bootcamp @ Ironhack Paris
* [Darewise](https://www.darewise.com/) - Developer behind Life Beyond
