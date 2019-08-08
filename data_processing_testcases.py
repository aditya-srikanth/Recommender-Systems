import csv
import pickle
from collections import defaultdict

movie_csv = open("movies.csv", encoding="utf8")
rating_csv = open("ratings.csv", encoding="utf8")

movie_reader = csv.DictReader(movie_csv)
rating_reader = csv.DictReader(rating_csv)

movie_dict = defaultdict(dict)
rating_dict = defaultdict(dict)

for row in movie_reader:
    movie_dict[row['movieId']] = {row['title']:row['genres'].split('|')}

for row in rating_reader:
    rating_dict[row['userId']][row['movieId']] = row['rating']

movie_size = 200           #INCLUSIVE OF 2000th movie
user_size = 150            #INCLUSIVE OF 1500th movie

new_movie_dict ={}
new_rating_dict ={}

#START PARAMETERS
start_movie = 1201
start_user = 1001


real_key = 1
for i in movie_dict.keys():
    movie_info = movie_dict[i]
    new_movie_dict[real_key] = movie_info
    for j in rating_dict.keys():
        rating_info = rating_dict[j]
        rating_keys = rating_info.keys()
        if i in rating_keys:
            rating = rating_info[i]
            del rating_info[i]
            rating_info[real_key] = rating
            new_rating_dict[j] = rating_info
    real_key = real_key + 1

final_movie_dict= {}
fresh_rating_dict ={}
final_rating_dict = {}
# for i in new_movie_dict.keys():
#     if(int(str(i),10) <= movie_size+start_movie and int(str(i),10) >=start_movie):
#         final_movie_dict[i] = new_movie_dict[i]

for i in new_rating_dict.keys():
    if(int(str(i),10) <= user_size+start_user and int(str(i),10) >=start_user):
        fresh_rating_dict[i] = new_rating_dict[i]

# for i in fresh_rating_dict.keys():
#     rating_list = fresh_rating_dict[i]
#     new_rating_list ={}
#     for j in rating_list.keys():
#         if(int(str(j),10) <= user_size+start_user and int(str(j),10) >=start_user):
#             new_rating_list[j] = rating_list[j]
#     final_rating_dict[i] = new_rating_list

# Testing via printing
for i in fresh_rating_dict.keys():
    print(i,fresh_rating_dict[i])

#Saving
# movie_dict_file = open("movie_file_test.txt", 'wb')
rating_dict_file = open("rating_file_test.txt", 'wb')

# pickle.dump(final_movie_dict, movie_dict_file)
pickle.dump(fresh_rating_dict ,rating_dict_file)

movie_csv.close()
rating_csv.close()
# movie_dict_file.close()
rating_dict_file.close()
