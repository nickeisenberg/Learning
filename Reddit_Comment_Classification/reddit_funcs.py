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

def dataset_makers(subs_coms_replies, fn):
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
