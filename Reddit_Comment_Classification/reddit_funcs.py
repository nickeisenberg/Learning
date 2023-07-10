import praw
import string
import numpy as np
import emoji
import random

# type(subreddit) = praw.models.reddit.subreddit.Subreddit
# sort_by: 'hot', 'new', 'top', 'rising'
# If search != None and type(search) = str then that will override sort_by
def submission_getter(subreddit=None,
                      sort_by='top',
                      search=None,
                      search_sort_by='relevance',
                      no_of_submissions=10):

    print('starting submission getter')

    if isinstance(search, str):
        submissions = subreddit.search(search, sort=search_sort_by)

    elif sort_by == 'top':
        submissions = subreddit.top(limit=no_of_submissions)
    elif sort_by == 'hot':
        submissions = subreddit.hot(limit=no_of_submissions)
    elif sort_by == 'new':
        submissions = subreddit.new(limit=no_of_submissions)
    elif sort_by == 'rising':
        submissions = subreddit.rising(limit=no_of_submissions)

    submission_list = []
    count = 1
    for sub in submissions:
        submission_list.append(sub)
        if count == no_of_submissions:
            break
        count += 1

    return submission_list


def comment_getter(submission_list=None,
                   no_of_comments=10):

    print('Getting comments for...')

    submission_coms = {submission: [] for submission in submission_list}

    for i, submission in enumerate(list(submission_coms.keys())):
        print(f'{i} / {len(submission_list)}: {submission.title}')
        submission_coms[submission] = submission.comments[: no_of_comments]

    return submission_coms


def comment_replies(submission_list=None,
                    submission_comments=None,
                    no_of_replies=10):

    print('starting comments replies')

    submissions_comments_replies = {sub: {} for sub in submission_list}

    for sub in submission_list:
        comments_replies = {com: [] for com in submission_comments[sub]}
        count_c = 1
        for com in submission_comments[sub]:
            print(f'COMMENT {count_c}')
            replies = com.replies
            replies.replace_more(limit=None)
            replies = replies[: no_of_replies]
            for reply in replies:
                comments_replies[com].append(reply)
            count_c += 1

        submissions_comments_replies[sub] = comments_replies

    return submissions_comments_replies

def dataset_makers_scr(subs_coms_replies, fn):
    with open(fn, 'w') as f:
        for isub, sub in enumerate(list(submissions[:1])):
            f.write(f"Title_{isub} -- {sub.title}")
            f.write('\n')
            for ic, com in enumerate(comments[sub]):
                f.write(f"Comment_{isub}.{ic} -- {com.body}")
                f.write('\n')
                for ir, rep in enumerate(subs_coms_reps[sub][com]):
                    f.write(f"Reply_{isub}.{ic}.{ir} -- {rep.body}")
                    f.write('\n')
    return None

def get_the_comments(subreddit,
                     search,
                     client_id,
                     client_secret,
                     user_agent):

    praw_sub = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent).subreddit(subreddit)

    # get the daily discussion submissions
    subs = submission_getter(subreddit=praw_sub,
                             search=search,
                             no_of_submissions=100)
    
    # get the comments
    coms = comment_getter(submission_list=subs,
                          no_of_comments=100)

    return coms

def dataset_makers_sc(com_dic, fn):
    with open(fn, 'w') as f:
        for s in com_dic.keys():
            f.write(f'!!post_title!!: {s.title}\n')
            for c in com_dic[s]:
                f.write(f'!!comment!!: {c.body}\n')
    return None


class TextProcessing:

    def __init__(self):
        self.words = {}
        self.word_counts = {}
        self.vocab = {}
        self.punc = string.punctuation + "‘’'“”"
        self.trans = str.maketrans('', '', string.punctuation + "‘’'“”")

    def strp_lower_rmpunc(
        self,
        input,
        from_file=False,
        to_file=False,
        to_file_path=None,
        return_output=True
        ):
        output = []
        if from_file:
            with open(input, 'r') as op:
                lines = op.readlines()
                for line in lines:
                    line = line.strip().lower().translate(self.trans)
                    output.append(line)
                    if to_file:
                        with open(to_file_path, 'a') as to_f:
                            to_f.write(line)
        else:
            input = input.strip().lower().translate(self.trans)
            if to_file:
                with open(to_file_path, 'a') as to_f:
                    to_f.write(input)
            output = input
        if return_output:
            return output

    def get_words(
        self,
        input,
        ignore=[],
        from_file=False,
        ):

        if from_file:
            with open(input, 'r') as op:
                lines = op.readlines()
                for line in lines:
                    line = line.strip().lower().translate(self.trans)
                    for word in line.split(' '):
                        if emoji.is_emoji(word) or word in ignore:
                            continue
                        if word not in self.words.keys():
                            self.words[word] = len(self.words)
                            self.word_counts[word] = 1
                        else:
                            self.word_counts[word] += 1

        else:
            for word in input:
                if emoji.is_emoji(word):
                    continue
                if word not in self.words.keys():
                    self.words[word] = len(self.words)
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1

        return None

    def get_vocab(self, no_words=10000):

        self.vocab['[UNK]'] = 0
        top_counts = np.array([*self.word_counts.values()])
        top_counts = top_counts[np.argsort(top_counts)][:: -1]
        cutoff_ind = min(no_words - 1, len(self.words) - 1)
        cutoff_val = top_counts[cutoff_ind]
        
        for word in self.words.keys():
            if self.word_counts[word] >= cutoff_val:
                if word not in self.vocab.keys():
                    self.vocab[word] = len(self.vocab) + 1
            if len(self.vocab) == 10000:
                break

class TextEncoding():

    def __init__(self, vocab):
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}

    def one_hot_encoding(self, line):
        one_hot = np.zeros(len(self.encoder))
        for word in line.split(' '):
            if word in self.encoder.keys():
                one_hot[self.encoder[word] - 1] = 1
            else:
                one_hot[0] = 1
        return one_hot

    def vectorize_encoding(self, line):
        vec = []
        for word in line.split(' '):
            if word in self.encoder.keys():
                vec.append(self.encoder[word])
            else:
                vec.append(0)
        vec = np.array(vec)
        return vec

    def vectorize_decoding(self, vec):
        line = []
        for v in vec:
            line.append(self.decoder[v])
        return line

class TextDataset:

    def __init__(self, x_dim, y_dim):
        self.x_train = np.zeros(x_dim) * np.nan
        self.y_train = np.zeros(y_dim) * np.nan
        self.x_val = np.zeros(x_dim) * np.nan
        self.y_val = np.zeros(y_dim) * np.nan
        self.x_test = np.zeros(x_dim) * np.nan
        self.y_test = np.zeros(y_dim) * np.nan

    def from_txt_file(
        self,
        encoder,
        path_to_txt_file,
        labels,
        shuffle=True,
        train_val_test_split=[.7, .15, .15]
        ):

        dataset_x = []
        with open(path_to_txt_file, 'r') as opt:
            lines = opt.readlines()
            for line in lines:
                dataset_x.append(encoder(line))
        dataset_x = np.array(dataset_x)
        
        if isinstance(labels, int):
            labels = np.repeat(labels, dataset_x.shape[0])

        inds = np.arange(dataset_x.shape[0])
        random.shuffle(inds)

        tr = int(train_val_test_split[0] * inds.size)
        vl = tr + int(train_val_test_split[1] * inds.size)

        self.x_train = np.vstack((self.x_train, dataset_x[: tr, :]))
        self.x_val = np.vstack((self.x_val, dataset_x[tr: vl, :]))
        self.x_test = np.vstack((self.x_test, dataset_x[vl:, :]))

        self.y_train = np.hstack((self.y_train, labels[inds][: tr]))
        self.y_val = np.hstack((self.y_val, labels[inds][tr: vl]))
        self.y_test = np.hstack((self.y_test, labels[inds][vl:]))

        return None

    def clean_up_nan(self):
        self.x_train = self.x_train[1:]
        self.y_train = self.y_train[1:]
        self.x_val = self.x_val[1:]
        self.y_val = self.y_val[1:]
        self.x_test = self.x_test[1:]
        self.y_test = self.y_test[1:]



 






