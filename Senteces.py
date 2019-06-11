class Sentences:

    def __init__(self):
        self.genre = {
            "movie_reviews" : [
                ['I loved the movie','pos'],
                ['I hated the movie','neg'],
                ['I liked the movie','pos'],
                ['I liked the actor','pos'],
                ['Loved the acting','pos'],
                ['Didn\'t liked the acting','neg'],
                ['concept of movie was good','pos'],
                ['movie eas conceptually bad','neg'],
                ['movie was not upto the par','neg'],
                ['stood upto the expectations','neu'],
                ['Actors worked hard for this movie','neu'],
                ['Poor acting','neg'],
                ['Too big','neg'],
                ['Best movie of the franchise','pos'],
                ['Loved ending scene','neu'],
                ['take a rating of neutral','neu'],
                ['expected a lot more','neg'],
                ['below par','neg'],
                ['Just at the edge','neu']
            ],

            "emails" : [
                ['Email is spam','spam'],
                ['Email is important','imp'],
                ['Spam alert','spam'],
                ['this has information','imp'],
                ['email contains junk','spam'],
                ['email is confidential','imp'],
                ['email has important information','imp'],
                ['email contains threats','spam'],
                ['email is urgent','imp'],
                ['this is urgent','imp'],
                ['it is from commercial sites','spam'],
                ['very important','imp'],
                ['from IMS','imp'],
                ['from Tnp Cell','imp'],
                ['email from facebook','spam'],
                ['email from social circle','spam']
            ]
        }

    def get_ds(self,name):
        return self.genre[name]

    def get_list(self):
        return self.genre.keys()
