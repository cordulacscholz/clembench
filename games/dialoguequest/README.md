# DialogueQuest: A Task-oriented Dialogue game based on the MultiWOZ dataset

### Game Description
The game consists of a dialogue between two players (the Questioner and the Answerer). Their goal is to reach a common goal that Player A is given. The goal is taken from one of the possible MultiWOZ domains (Hotel, Restaurant, Attraction). It consists of a set of constraints that need to be fulfilled (e.g. pricerange=cheap, area=east) and a set of requests that need to be fulfilled (e.g. get the phone number). Player B has a database at hand and is supposed to suggest a suitable item to Player A. Once Player A agrees on the suggestion, the game is finished.

### Instantiation
The game is based on the MultiWOZ corpus. The corpus files of the MultiWOZ2.2 version, which can be downloaded from the [MultiWOZ official repository](https://github.com/budzianowski/multiwoz), are used for the instance generation. For the current version of the game, only the setting of single-domain dialogues has been set uup, and the instances are restricted to the domains Hotel, Restaurant and Attraction, as those are the ones with the most items and can stand on their own, whereas the others often appear in multi-domain settings or are not suitable for natural dialogues. Those other domains, however, could be easily included into the implementation.  

### Implementation
In the current implementation, two sets of experiments are conducted: Firstly, there is a division into a reprompt option or not. Secondly, there is a division into the size of the database (5 or 20 items given).  
For each of the experiments we generate 10 instances with a randomly selected goal item from one of the three domains. The domain is selected in the order the domains are given in the constants file, resulting in all experiments having the same distribution of domains. To make sure that there is a viable solution available, 2-3 constraints and 2-3 requests are sampled from the goal object and then given to Player A in the initial prompt. Player B receives the sample from the domain database.  
One turn begins with an utterance of Player A in natural language and is followed by a reaction of Player B. The utterance is parsed if it terminates in a closing punctuation mark (for version EN). 
After this, Player B is prompted to sum up the contents of the two latest utterances (that of Player A and that of Player B) in a parsable json format. They are explicitly prompted to include either the id or the name of the item. If that can be found in the json structure, the given item is either added initally to an internal_state, or additional information about an item already in the current_state are added to that item. The game ends if Player A utters the fulfillment keyword. The final selection is parsed from Player A's last utterance, by matching either id or name. If no item can be detected, the final_suggestion (which serves for scoring) remains empty.

### Evaluation

**Turn-Level Scores**
Besides the general metrics of (passed and violated) requests count, some other datapoints are recorded per turn, namely the character and the word count for each player and the average sentence length for each player. A summarisation pentalty is recorded for each item in the json code that is not taken from the text to summarise (but from the original database instead). As this not in line with the original instruction, this adds to the summarisation penalty.  

**Episode-Level Scores**
An episode is considered successful if the task-success-rate (how many of the original constraints and the requests are fulfilled in the final suggestion) is >= 90%. The Main Score is the task-success rate - the summed up summarisation penalty over all turns.
Moreover, the word, character and sentence lengths are averaged and recorded. There is also a binary value respectively for either failure at generating proper text or proper json (if game is aborted).
