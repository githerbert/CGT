class Score:

    def __init__(self):

        self.scorelist = []

    def addScore(self, code_id, paper_id, sent_id ,score):
        my_tuple = (code_id, paper_id, sent_id, score)
        self.scorelist.append(my_tuple)