# DialogueQuest: A Task-oriented Dialogue game based on the MultiWOZ dataset

This game tests the abilities of ...

### Game Description
The game consists of a dialogue between two players (the Questioner and the Answerer). Their goal is to reach a goal that Player A is given. The goal is taken from one of the possible domains (Hotel, Restaurant, Attraction). It consists of constraints that need to be fulfilled (e.g. pricerange=cheap, area=east). Player B has a database at hand and is supposed to suggest a suitable item to Player A. Once Player A agrees on the suggestion, the game is finished.

### Instantiation
The game is based on the MultiWOZ corpus. The corpus files of the MultiWOZ2.2 version, which can be downloaded from the [MultiWOZ official repository](https://github.com/budzianowski/multiwoz), are used for the instance generation. For the current version of the game, only the setting of single-domain dialogues has been recreated, and the instances are restricted to the domains Hotel, Restaurant and Attraction, as those are the ones with the most items and can stand on their own, whereas the others often appear in multi-domain settings. Those other domains, however, could be easily included.  

### Implementation
For each of the experiments (reprompt allowed or not) we generate 10 instances with a randomly selected goal item from one of the three domains, which is also randomly slected. To make sure that there is a viable solution available, 2-3 constraints and 2-3 requests are sampled from the goal object and then given to Player A in the initial prompt. Player B receives a sample from the domain database (here: 15 items, with one of them being the goal item).  
One turn begins with an utterance of Player A in natural language and is followed by a reaction of Player B. The utterance is parsed if it terminates in a closing punctuation mark (for version EN). 
After this, Player B is prompted to sum up the contents of the two latest utterances (that of Player A and that of Player B) in a parsable json format. They are explicitly prompted to include either the id or the name of the item. If that can be found in the json structure, the given item is either added initally to an internal_state, or additional information about an item already in the current_state are added to that item. The game ends if Player A utters the fulfillment keyword. The final selection is parsed from Player A's last utterance, by matching either id or name. If no item can be detected, the final_suggestion (which serves for scoring) remains empty.

### Evaluation
The main goal of the game is reaching a solution which satisfies i) the constraints given and ii) the requests given. If the two players agree on an item in the end which is in line with those limiting factors to a certain degree (90%), the game is considered successful. The goal-fulfilling capability is therefore the main one tested.

**Turn-Level Scores**
Besides the general metrics of (passed and violated) requests count, some other datapoints are recorded per turn, namely the character and the word count for each player and the average sentence length for each player. They do not have an impact on the main score.

**Episode-Level Scores**
An episode is considered successful if the task-success-rate (how many of the original constraints and the requests are fulfilled in the final suggestion) is >= 90%. The main score ...  
Moreover, the word, character and sentence lengths are averaged and recorded. There is also a binary value respectively for either failure at generating proper text or proper json (if game is aborted).
