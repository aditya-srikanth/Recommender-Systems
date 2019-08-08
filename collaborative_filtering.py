'''This programn builds 3 recommender systems:
precision on top K, Spearman ranking and RMSE'''

import pickle
import math
import scipy.stats as ss
import numpy
import random

if __name__ == "__main__":
    print("Unpickling data...")
    movie_pickle = open("movie_file.txt", 'rb')
    rating_pickle = open("rating_file.txt", 'rb')

    movie_dict = pickle.load(movie_pickle)
    rating_dict  = pickle.load(rating_pickle)

    movieIds = movie_dict.keys()
    userIds = rating_dict.keys()
    last_movieId = int(list(movieIds)[-1])

    user_rating_matrix = [0] * (len(userIds) + 1)

    for i in range(0, len(user_rating_matrix)):
        user_rating_matrix[i] = [0] * (last_movieId + 1)

    print("Creating user-rating matrix...")
    for user in userIds:
        user_movies = rating_dict[user].keys()
        for movie in user_movies:
            user_rating_matrix[int(user)][int(movie)] = float(rating_dict[user][movie])

    # Baseline estimation
    print("Calculating mean ratings of users...")
    mean_user_rating_dict = {}
    length = 0
    total_rating = 0
    for user in userIds:
        total_user_rating = 0
        temp_length = 0
        for movie in movieIds:
            if user_rating_matrix[int(user)][int(movie)] > 0:
                total_user_rating = total_user_rating + user_rating_matrix[int(user)][int(movie)]
                temp_length = temp_length + 1
        total_rating = total_rating + total_user_rating
        length = length + temp_length
        if temp_length > 0:
            mean_user_rating = round(total_user_rating / temp_length, 2)
            mean_user_rating_dict[user]  = str(mean_user_rating)

    print("Calculating mean ratings of movies...")
    mean_movie_rating_dict = {}
    for movie in movieIds:
        movie_rating = 0
        temp_length = 0
        for user in userIds:
            if user_rating_matrix[int(user)][int(movie)] > 0:
                movie_rating = movie_rating + user_rating_matrix[int(user)][int(movie)]
                temp_length = temp_length + 1
        if temp_length > 0:
            mean_movie_rating = round(movie_rating / temp_length, 2)
        else:
            mean_movie_rating = 0
        mean_movie_rating_dict[str(movie)] = str(mean_movie_rating)

    total_mean_rating = round(total_rating / length, 2)

    #User input
    method = int(input("Enter method: 1. K-neigbours 2. Spearman Ranking 3. RMSE: "))
    if method != 3:
        test_user = int(input("Enter userId: "))
        test_user_movies_dict = rating_dict[str(test_user)]
        test_user_movies = test_user_movies_dict.keys()
        mean_test_rating = 0
        test_user_rating = 0
        last_movie = int(list(rating_dict[str(test_user)].keys())[-1]) + 1

    # K-neighbours method
    if method == 1:
        test_user_rating_dict = rating_dict[str(test_user)]
        sorted_test_rating_dict = {r: test_user_rating_dict[r] for r in sorted(test_user_rating_dict, key=test_user_rating_dict.get, reverse=True)}
        top_test_movies = list(sorted_test_rating_dict.keys())
        top_K = top_test_movies[0:5]
        actual_ratings = []
        pred_ratings = []
        pred_ratings_baseline = []
        print("Predicting ratings for top K ratings...")
        for test_movie in top_K:
            test_movie = int(test_movie)
            test_baseline = total_mean_rating + (float(mean_user_rating_dict[str(test_user)]) - total_mean_rating) + (float(mean_movie_rating_dict[str(test_movie)]) - total_mean_rating)
            pearson_dict = {}
            # print("Calculating similarities...")
            for user in userIds:
                user = int(user)
                user_rating = 0
                temp_numerator = 0
                temp_denominator_x = 0
                temp_denominator_y = 0
                length = 0
                for movie in range(0, last_movie):
                    if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                        test_user_rating = test_user_rating + user_rating_matrix[test_user][movie]
                        user_rating = user_rating = user_rating + user_rating_matrix[user][movie]
                        length = length + 1
                if length > 0:
                    mean_test_user_rating = test_user_rating / length
                    mean_user_rating = user_rating / length
                    for movie in range(1, last_movie):
                        if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                            temp_numerator = temp_numerator + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) * (user_rating_matrix[user][movie] - mean_user_rating))
                            temp_denominator_x = temp_denominator_x + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) ** 2)
                            temp_denominator_y = temp_denominator_y + ((user_rating_matrix[user][movie] - mean_user_rating) ** 2)
                    temp_denominator = math.sqrt(temp_denominator_x) * math.sqrt(temp_denominator_y)
                    if temp_denominator > 0:
                        coeff = temp_numerator / temp_denominator
                        pearson_dict[user] = coeff

            sorted_pearson_dict = {t: pearson_dict[t] for t in sorted(pearson_dict, key=pearson_dict.get, reverse=True)}

            top_matches = {k:sorted_pearson_dict[k] for k in list(sorted_pearson_dict)[:5]}   # Taking 5 nearest neighbours

            top_users = list(top_matches.keys())

            temp_numerator = 0
            temp_denominator = 0
            temp_numerator_baseline = 0

            # print("Predicting rating...")
            for user in top_users:
                if top_matches[user] != 1:
                    user_baseline = float(mean_user_rating_dict[str(user)]) - user_rating_matrix[user][test_movie]
                    temp_numerator_baseline = temp_numerator_baseline + (float(top_matches[user]) * (user_rating_matrix[user][test_movie] - user_baseline))
                    temp_numerator = temp_numerator + (float(top_matches[user]) * user_rating_matrix[user][test_movie])
                    temp_denominator = temp_denominator + float(top_matches[user])

            pred_rating = abs(round(temp_numerator / temp_denominator, 2))
            pred_rating_baseline = abs(round(temp_numerator_baseline / temp_denominator, 2))
            test_rating = user_rating_matrix[test_user][test_movie]

            pred_ratings.append(pred_rating)
            pred_ratings_baseline.append(pred_rating_baseline)
            actual_ratings.append(test_rating)
        pres_count = 0
        pres_count_baseline = 0
        for i in range(0, len(actual_ratings)):
            if actual_ratings[i] >= 3.5:
                if pred_ratings[i] >= 3.5:
                    pres_count = pres_count + 1
                if pred_ratings_baseline[i] >= 3.5:
                    pres_count_baseline = pres_count_baseline + 1
            if actual_ratings[i] < 3.5:
                if pred_ratings[i] < 3.5:
                    pres_count = pres_count + 1
                if pred_ratings_baseline[i] < 3.5:
                    pres_count_baseline = pres_count_baseline + 1
        precision = round((pres_count / len(actual_ratings)) * 100, 2)
        precision_baseline = round((pres_count_baseline / len(actual_ratings)) * 100, 2)
        print("Precision of top K: " + str(precision) + "%")
        print("Precision of top K (baseline): " + str(precision_baseline) + "%")

    # Spearman Method
    elif method == 2:
        test_movie = int(input("Enter movieId: "))
        test_baseline = total_mean_rating + (float(mean_user_rating_dict[str(test_user)]) - total_mean_rating) + (float(mean_movie_rating_dict[str(test_movie)]) - total_mean_rating)
        temp_test_rating_dict = {}
        temp_user_rating_dict = {}
        for user in userIds:
            user = int(user)
            some_list_1 = []    # for test user
            some_list_2 = []    # for other user
            for movie in range(0, last_movie):
                if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                    some_list_1.append((str(user_rating_matrix[test_user][movie])))
                    some_list_2.append(str(user_rating_matrix[user][movie]))
            temp_test_rating_dict[str(user)] = some_list_1
            temp_user_rating_dict[str(user)] = some_list_2

        test_movies_ranks = {}
        user_movies_ranks = {}

        print("Calculating ranks...")
        for user in temp_user_rating_dict.keys():
            test_movies_ranks[user] = ss.rankdata([-1 * user for user in temp_test_rating_dict[user]])   # Check code later
            user_movies_ranks[user] = ss.rankdata([-1 * user for user in temp_user_rating_dict[user]])   # Check here also

        spearman_dict = {}
        users = test_movies_ranks.keys()

        print("Calculating similarities...")
        for user in users:
            sq_d = 0
            if len(temp_test_rating_dict[user]) > 0:
                for rank in range(0, len(temp_test_rating_dict[user])):
                    sq_d = sq_d + (test_movies_ranks[user][rank] - user_movies_ranks[user][rank]) ** 2
                result = 1 - ((6 * sq_d) / (len(temp_test_rating_dict[user]) * ((len(temp_test_rating_dict[user]) ** 2) - 1)))
                spearman_dict[user] = str(result)

        temp_numerator = 0
        temp_denominator = 0
        temp_numerator_baseline = 0

        for user in users:
            if len(temp_test_rating_dict[user]) > 0:
                if user is not str(test_user) and abs(float(spearman_dict[user])) > 0.35:
                    user_baseline = float(mean_user_rating_dict[user]) - user_rating_matrix[int(user)][test_movie]
                    temp_numerator_baseline = temp_numerator_baseline + (float(spearman_dict[user]) * (user_rating_matrix[int(user)][test_movie] - user_baseline))
                    temp_numerator = temp_numerator + (float(spearman_dict[user]) * user_rating_matrix[int(user)][test_movie])
                    temp_denominator = temp_denominator + float(spearman_dict[user])

        print("Predicting rating...")
        pred_rating = abs(round((temp_numerator / temp_denominator), 2))
        pred_rating_baseline = abs(round(temp_numerator_baseline / temp_denominator, 2))
        test_rating = user_rating_matrix[test_user][test_movie]
        print("Predicted rating: " + str(pred_rating))
        print("Predicted rating (baseline): " + str(pred_rating_baseline))
        if test_rating > 0:
            spearman_error = abs(pred_rating - test_rating) * (100 / (test_rating))
            spearman_error_baseline = abs(pred_rating_baseline - test_rating) * (100 / test_rating)
            print("Actual rating: " + str(test_rating))
            print("Closeness for Spearman ranking: " + str(100 - round(spearman_error, 2)) + "%")
            print("Closeness for Spearman ranking (baseline): " + str(100 - round(spearman_error_baseline, 2)) + "%")

    # Root Mean Square Error
    elif method == 3:
        test_users = list(numpy.random.randint(1, len(userIds), size=int(0.2*len(userIds))))
        errors = []
        errors_baseline = []
        movieIds = list(movieIds)

        for test_user in test_users:
            test_user = int(test_user)
            print("##### Test user: " + str(test_user))
            test_movie = random.randint(1, len(movieIds))
            test_movie = movieIds[test_movie]
            test_movie = int(test_movie)
            print("##### Test movie: " + str(test_movie))
            last_movie = int(list(rating_dict[str(test_user)].keys())[-1]) + 1
            test_baseline = total_mean_rating + (float(mean_user_rating_dict[str(test_user)]) - total_mean_rating) + (float(mean_movie_rating_dict[str(test_movie)]) - total_mean_rating)
            pearson_dict = {}
            mean_test_rating = 0
            test_user_rating = 0
            print("Calculating similarities...")
            for user in userIds:
                user = int(user)
                user_rating = 0
                temp_numerator = 0
                temp_denominator_x = 0
                temp_denominator_y = 0
                length = 0
                for movie in range(0, last_movie):
                    if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                        test_user_rating = test_user_rating + user_rating_matrix[test_user][movie]
                        user_rating = user_rating = user_rating + user_rating_matrix[user][movie]
                        length = length + 1
                if length > 0:
                    mean_test_user_rating = test_user_rating / length
                    mean_user_rating = user_rating / length
                    for movie in range(1, last_movie):
                        if (user_rating_matrix[test_user][movie] > 0) and (user_rating_matrix[user][movie] > 0):
                            temp_numerator = temp_numerator + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) * (user_rating_matrix[user][movie] - mean_user_rating))
                            temp_denominator_x = temp_denominator_x + ((user_rating_matrix[test_user][movie] - mean_test_user_rating) ** 2)
                            temp_denominator_y = temp_denominator_y + ((user_rating_matrix[user][movie] - mean_user_rating) ** 2)
                    temp_denominator = math.sqrt(temp_denominator_x) * math.sqrt(temp_denominator_y)
                    if temp_denominator > 0:
                        coeff = temp_numerator / temp_denominator
                        pearson_dict[user] = coeff

            sorted_pearson_dict = {t: pearson_dict[t] for t in sorted(pearson_dict, key=pearson_dict.get, reverse=True)}

            top_matches = {k:sorted_pearson_dict[k] for k in list(sorted_pearson_dict)[:5]}

            top_users = list(top_matches.keys())

            temp_numerator = 0
            temp_denominator = 0
            temp_numerator_baseline = 0

            print("Predicting rating...")
            for user in top_users:
                if top_matches[user] != 1:
                    user_baseline = float(mean_user_rating_dict[str(user)]) - user_rating_matrix[user][test_movie]
                    temp_numerator_baseline = temp_numerator_baseline + (float(top_matches[user]) * (user_rating_matrix[user][test_movie] - user_baseline))
                    temp_numerator = temp_numerator + (float(top_matches[user]) * user_rating_matrix[user][test_movie])
                    temp_denominator = temp_denominator + float(top_matches[user])

            if temp_denominator > 0:
                pred_rating = abs(round(temp_numerator / temp_denominator, 2))
                pred_rating_baseline = abs(round(temp_numerator_baseline / temp_denominator, 2))
                test_rating = user_rating_matrix[test_user][test_movie]
                error = (pred_rating - test_rating) ** 2
                error_baseline = (pred_rating_baseline - test_rating) ** 2
                errors.append(round(error, 2))
                errors_baseline.append(round(error_baseline, 2))

        rms_error = round(math.sqrt((round((sum(errors) / len(errors)), 2))), 2)
        rms_error_baseline = round(math.sqrt(round((sum(errors_baseline) / len(errors_baseline)), 2)), 2)
        print("RMSE of recommender system: " + str(rms_error))
        print("RMSE of recommender system (baseline): " + str(rms_error_baseline))
