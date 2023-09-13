import numpy as np
import math
def generate_rrsd_matrix(n, d, delta):

    p = np.exp(-1/d)
    r = int((1-p)*(n-d+1))
    print('Tang:r is {}\n'.format(r))
    m = int(np.exp(1)*d*math.log(2*n/delta))
    print('Tang: the number of tests required is {}\n'.format(m))
    matrix = []

    matrix = np.zeros((m, n))
    for i in range(m):
        # Randomly select r columns to set to 1 for each row
        random_columns = np.random.choice(n, r, replace=False)
        matrix[i, random_columns] = 1

    matrix = np.array(matrix)
    return matrix


test_matrix = generate_rrsd_matrix(10000,5,0.01)
print(test_matrix)
'''
def algorithm_4_wrapper(
    pkl_name:str, 
    exp_title:str, 
    each_method_GigaMACs:float, # GMacs per test. M images in total. 
    group_size:int, # M value in the paper 
    confidence_threshold:float=0.5,
    pkl_dir='./prediction_cache_0.1/', # default root dir 
    ):
    print("##################################")
    print(exp_title)
    method_score, method_target = load_validate_dump(pkl_name=pkl_name, pkl_dir=pkl_dir, verbose=True, confidence_threshold=confidence_threshold)
        
    method_tests_Round_1 = len(method_target)
    num_firearm_samples = np.sum(method_target)
    r=8
    delta=0.1

    method_TeraMACs_Round_1 = each_method_GigaMACs / 1000 * method_tests_Round_1 # TMacs 10^12
    print("Number of Tests (1st Round): ", method_tests_Round_1, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_1))

    # Generate RrSD matrix
    rrsd_matrix = generate_rrsd_matrix(len(method_score), num_firearm_samples, r, delta)  # Assuming r is globally defined or passed as a parameter

    # Reshape method_score based on RrSD matrix
    grouped_scores = []
    for row in rrsd_matrix:
        group = method_score[row == 1]
        grouped_scores.append(group)

    method_Round_1_next = []
    for group in grouped_scores:
        avg_score = np.mean(group)
        # Determine the boolean value for this group based on the average score
        group_value = 1 if avg_score > confidence_threshold else 0
        # Append the boolean value for each sample in the group
        method_Round_1_next.extend([group_value] * r)  # r is the size of the group, which is fixed (8 or 16 in your case)
    
    print("Number Of Samples After the 1st round:", np.sum(method_Round_1_next))
    ##################################
        # Algorithm 4 comes in here 
        # Insert a Round-2
    ##################################
        if group_size == 8:
            # scheme 1: M=8, 4 K1G2 (ResNeXt101FullK7_imgnet_G2K1.pkl) + 2 base (vary with positives in K1G2)
            # candidates: ResNeXt101FullK7_imgnet_G2.pkl, ResNeXt101FullK7_TREE042_G2.pkl, ResNeXt101FullK7_TREE024_G2.pkl
            # 2nd-level: use K1G2
            each_2nd_GigaMACs = 20.16 - 7.3 # could minus the base feature extraciton 
            group_size_2nd = 2 
            method_2nd_level_score, _ = load_validate_dump(pkl_dir=pkl_dir, pkl_name='ResNeXt101FullK7_imgnet_G2K1.pkl', verbose=False, confidence_threshold=0.5)

        elif group_size == 16:
            # scheme 2: M=16, 4 K3G2 (ResNeXt101FullK7_imgnet_G2K3.pkl) + 4 base (vary with positives in K1G2)
            # candidates: ResNeXt101FullK7_imgnet_G2K15.pkl, ResNeXt101FullK7_TREE024_G2_028.pkl 
            # 2nd-level: use K3G2
            each_2nd_GigaMACs = 27.46 - 14.6 # could minus the base feature extraciton 
            group_size_2nd = 4 
            method_2nd_level_score, _ = load_validate_dump(pkl_dir=pkl_dir, pkl_name='ResNeXt101FullK7_imgnet_G2K3.pkl', verbose=False, confidence_threshold=0.5)

        else:
            raise NotImplementedError() 

        method_2nd_level_score_repeat = np.repeat(method_2nd_level_score, group_size_2nd)
        method_Round_2_next = np.logical_and(method_Round_1_next, method_2nd_level_score_repeat>0.5)
        method_tests_Round_2 = np.sum(method_Round_1_next) // group_size_2nd # div group size second level 
        method_TeraMACs_Round_2 = each_2nd_GigaMACs / 1000 * method_tests_Round_2 # TMacs 10^12
        print("Number of Tests (2nd Round): ", method_tests_Round_2, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_2))

        ##################################
        # Finish Round-2. Comes Round-3. 
        ##################################

        K0_score, K0_target = load_validate_dump(pkl_dir=pkl_dir, pkl_name="ResNeXt101FullK0.pkl", verbose=False)

        method_recall = 100 * np.sum( np.logical_and(
            np.logical_and(K0_target, K0_score>0.5), 
            method_Round_2_next) ) / np.sum(K0_target==1) # use K0 model as the second round 
        method_FPR = 100 * np.sum(   np.logical_and(
            np.logical_and(K0_target==0, K0_score>0.5), 
            method_Round_2_next) 
            ) / np.sum(K0_target==0) # False Positive Rate 
        print("Recall(%): {} FPR(%): {:3f}".format(method_recall, method_FPR))


        method_tests_Round_3 = np.sum(method_Round_2_next) 
        each_K0_GigaMACs = 16.5 # 16.5 GMacs per test, same as the baseline model 
        method_TeraMACs_Round_3 = each_K0_GigaMACs / 1000 * method_tests_Round_3 # TMacs 10^12
        print("Number of Tests (3rd Round): ", method_tests_Round_3, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_3))

        method_TeraMACs_total = method_TeraMACs_Round_1 + method_TeraMACs_Round_2 + method_TeraMACs_Round_3
        method_tests_total = method_tests_Round_1 + method_tests_Round_2 + method_tests_Round_3
        print("Total Computation: {:.1f} TeraMACs".format(method_TeraMACs_total), "Total Tests:", method_tests_total, "Relative Cost", method_TeraMACs_total/805.2)

        result_dict = {
            'method_score': method_score, # raw outputs 
            'method_target': method_target, # raw outputs 
            'method_recall': method_recall, # performance metrics 
            'method_FPR': method_FPR, # performance metrics 
            'method_tests_Round_1': method_tests_Round_1, # computation cost metrics
            'method_tests_Round_3': method_tests_Round_3, # computation cost metrics
            'method_TeraMACs_Round_1': method_TeraMACs_Round_1, # computation cost metrics
            'method_TeraMACs_Round_3': method_TeraMACs_Round_3, # computation cost metrics
            'method_TeraMACs_total': method_TeraMACs_total, # computation cost metrics
        }
        return result_dict

    K7G2_A2_result_dict = algorithm_4_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2.pkl', 
        exp_title='K=7 + Design 2 (G2) + Algorithm 2 Three-Round', 
        each_method_GigaMACs=42.06, 
        group_size=8,
        )


    TREE024_result_dict = algorithm_4_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2.pkl', 
        exp_title='Design 3 (Tree024) + K=7 + Algorithm 2 Three-Round', 
        each_method_GigaMACs=33.25, 
        group_size=8,
        # confidence_threshold=0.8
        )

    # Group Size 16 
    K15G2_result_dict = algorithm_4_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2K15.pkl', 
        exp_title='K=15 + Design 2 (G2) + Algorithm 2 Three-Round', 
        each_method_GigaMACs=71.27, 
        group_size=16,
        )


    TREE028_result_dict = algorithm_4_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2_028.pkl', 
        exp_title='Design 3 (Tree028) + K=7 + Algorithm 2 Three-Round', 
        each_method_GigaMACs=53.65, 
        group_size=16,
        )

    print("##################################")
'''