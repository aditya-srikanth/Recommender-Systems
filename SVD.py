import pickle
import math
import numpy as np
import random
import time
np.set_printoptions(threshold=np.inf)


def Energy(A):
    """
    This function calculates the energy of the matrix.

    @type  A: Square matrix (numpy Array)
    @param A: Matrix for which energy must be calculated
    @rtype:   number
    @return: Energy of the matrix
    """
    A = A*A
    return A.sum()


def RMSE(user_array,FinalA):
    """
    This function calculates the Root-Mean-Square-Error value obtained between two matrices

    @type  user_array: Square matrix (numpy Array)
    @param user_array: The original matrix before SVD decomposition
    @type  FinalA: Square matrix (numpy Array)
    @param FinalA: The matrix after SVD decomposition
    @rtype:  number
    @return: Root-Mean-Square-Error value obtained
    """
    error = user_array - FinalA
    # error = error[1:,:]
    sqerror = error*error
    # print(sqerror.size)
    RMSE = sqerror.sum()/(sqerror.size)
    RMSE = math.sqrt(RMSE)
    return RMSE

def SVD(user_array):
    """
    This function calculates the SVD decomposition of a given matrix

    @type  user_array: Square matrix (numpy Array)
    @param user_array: The original matrix before SVD decomposition
    @rtype: Tuple of (Square matrix,Square matrix,Square matrix,Square matrix)
    @return: Tuple of U , sigma , V ,& the obtained eigenvalues
    """
    UAT = user_array.T
    array_AAT = np.dot(user_array, (UAT))
    eigenvalues, eigenvectors_AAT = np.linalg.eig(array_AAT)
    eigenvectors_AAT = eigenvectors_AAT.real
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_AAT = eigenvectors_AAT[:,idx]

    eigenvalues[eigenvalues < 1.0e-10] = 0
    #Finding rank
    rank = 0
    for i in eigenvalues:
        rank = rank + 1
        if(i.imag != 0):
            break
    rank = rank -1
    print("Rank: ",rank)
    # Rank = Row that contains last non zero value
    #Reducing size of eigenvalues to only include the actual rank
    eigenvalues = eigenvalues[0:(rank-1)]
    # eigenvalues = eigenvalues.real
    #Build sigma
    sigma = np.diag(eigenvalues)
    #U of SVD with
    U = eigenvectors_AAT
    array_ATA = np.dot((UAT), user_array)
    eigenvalues_irr, eigenvectors_ATA = np.linalg.eig(array_ATA)
    idx = eigenvalues_irr.argsort()[::-1]
    eigenvalues_irr = eigenvalues_irr[idx]
    eigenvectors_ATA = eigenvectors_ATA[:,idx]
    V = eigenvectors_ATA
    # Slicing to match Size
    U = U[:, 0:rank-1]
    V = V[:, 0:rank-1]
    # print(sigma)
    sigma = np.sqrt(sigma)
    print("Size of U,V",U.shape, V.shape)
    return U,sigma,V,eigenvalues

def Query(q,V):
    """
    This function queries the SVD matrix given a query vector

    @type  q: Square matrix (1D) (numpy Array)
    @param q: Query vector
    @type  V: Square matrix (numpy Array)
    @param V: The V obtained from the SVD
    @rtype: Tuple (Square matrix (1D) (numpy Array),number)
    @return: Tuple (The result vector obtained,time taken)
    """
    start_time = time.clock()
    temp = np.dot(q,V)
    final = np.dot(temp,V.T)
    duration = time.clock() - start_time
    # print(duration)
    return final,duration

def Precision_top_k(k,q,final):
    """
    This function calculates the Precision Top K

    @type  k : number
    @param k : The k in Precision Top k
    @type q: Square matrix (1D) (numpy Array)
    @para q: Query Vector
    @type  final: Square matrix (1D) (numpy Array)
    @param final: Final matrix obtained from Query
    @rtype:  number
    @return: Precision Top K value obtained
    """

    # print("F,q Shape",final.shape,q.shape)
    final[final < 3.5] = 0
    final[final > 3.5] = 1
    q[q < 3.5] = 0
    q[q > 3.5] = 1
    idx = final.argsort()[::-1]
    final = final[idx]
    q = q[idx]
    prec_val = 0
    # for i in range(0,final.shape[0]):
    #     print(final[i],q[i])
    for i in range(0,k-1):
        if((final[i] == 1 and q[i] == 1) or (final[i] == 0 and q[i] == 0)):
            prec_val +=1
    prec_val = prec_val / k
    return prec_val

def spearmanCoefficient(predicted_rating,test_rating):
    """
    This function calculates the Spearman Coefficient

    @type  predicted_rating: Square matrix (1D) (numpy Array)
    @param predicted_rating: The Predicted rating obtained through Querying
    @type  test_rating: Square matrix (1D) (numpy Array)
    @param test_rating: The actual list of ratings given by a user
    @rtype:  number
    @return: Spearman Coefficient obtained
    """
    predicted_rank = np.argsort(predicted_rating)
    test_rank = np.argsort(test_rating)
    d = test_rank - predicted_rank
    d_squared = np.power(d,2)
    sum_d_squared = np.sum(d_squared)
    n = d.shape[0]
    rho = 1 - (6*sum_d_squared)/(n*(n**2 - 1))
    return rho


if __name__ == "__main__":
    movie_size = 2000  # INCLUSIVE OF 2000th movie
    user_size = 610  # INCLUSIVE OF 1500th movie
    test_shift = 10

    movie_pickle = open("movie_file.txt", 'rb')
    rating_pickle = open("rating_file.txt", 'rb')

    movie_dict = pickle.load(movie_pickle)
    rating_dict = pickle.load(rating_pickle)

    movieIds = movie_dict.keys()
    userIds = rating_dict.keys()

    user_rating_matrix = [0] * (len(userIds) + 1)

    for i in range(0, len(user_rating_matrix)):
        user_rating_matrix[i] = [0] * (movie_size + 1)  # Possible Change

    for user in userIds:
        user_movies = rating_dict[user].keys()
        for movie in user_movies:
            user_rating_matrix[int(user)][int(movie)] = float(
                rating_dict[user][movie])

    user_array_store = np.array(user_rating_matrix)
    test_array = user_array_store[user_size-test_shift:(user_size),:]
    user_array = user_array_store[1:(user_size-test_shift),:]

    U,sigma,V,eigenvalues = SVD(user_array)

    VT = V.T
    # U = U.real
    # V = V.real
    new_A = np.dot(U,sigma)
    FinalA = np.dot(new_A,VT)

    print(FinalA.shape)

    energy = Energy(eigenvalues)
    # Reverse the eigenvalue np array
    Reduction_array = np.empty([1])
    for i in range(eigenvalues.size,0,-1):
        temp = eigenvalues[0:i]
        temp_Energy = Energy(temp)
        if(temp_Energy >= 0.9 * energy):
            Reduction_array = temp
        else:
            break

    #90% Reduced Matrix size
    size = Reduction_array.size
    print(size)
    Reduction_array = Reduction_array[0:(size-1)]
    Reduction_array = Reduction_array.real
    #Remake the sigma
    sigma_reduced = np.diag(Reduction_array)
    U_reduced = U[:,0:(size-1)]
    V_reduced = V[:,0:(size-1)]
    VT_reduced = V_reduced.T
    new_A_reduced = np.dot(U_reduced,sigma_reduced)
    ReducedA = np.dot(new_A_reduced,VT_reduced)


    print("Non reduced",RMSE(user_array,FinalA))
    print("90% reduced",RMSE(user_array,ReducedA))

    # randvar =random.randint(0,1000)
    # q = user_array[randvar,:]
    # print(test_array.shape[0])
    psum = 0
    psum_red = 0
    scsum = 0
    scsum_red = 0
    dur_sum =0
    dur_red_sum =0
    for i in range(1,test_array.shape[0]):
        q = test_array[i,:]
        # print(q)
        predicted_rating,dur = Query(q,V)
        predicted_rating_reduced,dur_red = Query(q,V_reduced)
        dur_sum += dur
        dur_red_sum += dur_red
        psum += Precision_top_k(10,q,predicted_rating)
        psum_red += Precision_top_k(10,q,predicted_rating_reduced)
        scsum += spearmanCoefficient(predicted_rating,q)
        scsum_red += spearmanCoefficient(predicted_rating_reduced,q)

    print("Precision_top_10: ",psum/test_array.shape[0])
    print("Precision_top_10 (90% reduced): ",psum_red/test_array.shape[0])
    print("Spearman Coeff:",scsum/test_array.shape[0])
    print("Spearman Coeff (90% reduced):",scsum_red/test_array.shape[0])
    print("Duration:",dur_sum/test_array.shape[0])
    print("Duration (90% reduced):",dur_red_sum/test_array.shape[0])

    U_file = open("U_file.txt", 'wb')
    V_file = open("V_file.txt", 'wb')
    sigma_file = open("sigma_file.txt", 'wb')
    U_reduced_file = open("U_reduced_file.txt", 'wb')
    V_reduced_file = open("V_reduced_file.txt", 'wb')
    sigma_reduced_file = open("sigma_reduced_file.txt", 'wb')

    user_map = np.dot(U,sigma)
    sigma_map = np.dot(sigma,V.T)



    pickle.dump(U, U_file)
    pickle.dump(V, V_file)
    pickle.dump(sigma ,sigma_file)

    U_file.close()
    V_file.close()
    sigma_file.close()

    movie_pickle.close()
    rating_pickle.close()

    pickle.dump(U_reduced, U_reduced_file)
    pickle.dump(V_reduced, V_reduced_file)
    pickle.dump(sigma_reduced ,sigma_reduced_file)

    U_reduced_file.close()
    V_reduced_file.close()
    sigma_reduced_file.close()
