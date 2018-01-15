class Score:

    def __init__(self):

        self.scorelist = []

    def addScore(self, code_id, paper_id, score):
        self.scorelist.append(code_id,paper_id,score)