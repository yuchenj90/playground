import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def parse_transcript(transcript):
    regex_user_group = r'(&&& (user) &&& \(([A-Z0-9:\s-]*)\):)'
    regex_user = r'&&& user &&& \([A-Z0-9:\s-]*\):'
    regex_agent_group = r'(\/\/\/ ([A-Za-z\s0-9]+) \/\/\/ \(([A-Z0-9:\s-]*)\):)'
    regex_agent = r'\/\/\/ [A-Za-z\s0-9]+ \/\/\/ \([A-Z0-9:\s-]*\):'
    regex_system = r'system \([A-Z0-9:\s-]*\):'
    # Step 1: get dialog with role, ts, msg
    dialog = []
    user_groups = re.findall(regex_user_group, transcript)
    for i, x in enumerate(re.split(regex_user, transcript)[1:]):
        agent_groups = re.findall(regex_agent_group, x)
        for j, msg in enumerate(re.split(regex_agent, x)):
            if j == 0:  # get user message
                pos = re.search(regex_system, msg)
                if pos:
                    msg = msg[:pos.start()]    
                dialog.append([user_groups[i][1], user_groups[i][2], msg.strip()])
            else:  # get agent messages
                dialog.append([agent_groups[j-1][1], agent_groups[j-1][2], msg.strip()])
    # Step 2: parse user question, agent first response and agent action
    user_question = ''
    agent_first_response = {'msg': '', 'action': ''}
    agent_first_turn_msgs = ''
    agent_msgs = []
    regex_action = r'\n- Agent action to suggested response: (.*)'
    regex_multiturn = r'\n- Multi-turn SR: (.*)'
    datestr, ts = None, None
    for i in range(len(dialog)):
        conv = dialog[i]
        if conv[0] == 'user' and len(user_question)==0: 
            user_question = conv[2]
            datestr = conv[1][:8]
            ts = conv[1][:20]
        if conv[0] != 'user' and 'Suggested response mode: greeting' not in conv[2]:
            action = re.findall(regex_action, conv[2])
            multiturn = re.findall(regex_multiturn, conv[2])
            if action and len(multiturn)>0 and multiturn[0].lower()=='false':
                agent_first_response['action'] = action[0]
            agent_first_response['msg'] = re.split(regex_action, conv[2])[0]
            agent_first_turn_msgs = re.split(regex_action, conv[2])[0]
            pos = i+1
            while pos < len(dialog):
                if dialog[pos][0] != 'user':
                    agent_first_turn_msgs += ('\n' + dialog[pos][2]) 
                else:
                    break
                pos += 1
            break
    for i in range(len(dialog)):
        conv = dialog[i]
        if conv[0] != 'user':
            agent_msgs.append(re.split(regex_action, conv[2])[0])
            pos = i+1
            while pos < len(dialog):
                if dialog[pos][0] != 'user':
                    agent_msgs.append(dialog[pos][2]) 
                pos += 1

    res = {'datestr': datestr, 
           'ts': ts,
           'user_question': user_question, 
           'agent_first_msg': agent_msgs[1] if len(agent_msgs)>1 else "", 
           'agent_action': agent_first_response['action'],
           'agent_first_turn_msgs': agent_first_turn_msgs,
           'agent_msgs': agent_msgs,
           }
    return dialog, res