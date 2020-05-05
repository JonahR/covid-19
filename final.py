import math
import random

def get_dataset(pre):
    pre_stay_at_home_cases = [6,10,19,23,27,
    32,49,61,61,72,
    88,114,133,158,172,
    183,194,215,228,244,
    253,269,278,289,302]

    post_stay_at_home_cases = [307,313,327,334,340,344,351,352,351,358,361,364,381,386,387,395,401,405,412,417,424,428,430,433,436,441]


    pre_data = list()
    for i, cases in enumerate(pre_stay_at_home_cases):
        pre_data.append([i, math.log(cases)])

    post_data = list()
    for i, cases in enumerate(post_stay_at_home_cases):
        post_data.append([i, math.log(cases)])

    if pre:
        return pre_data
    else:
        return post_data


def regression(pre, beta_0, beta_1):
    MSE = 0.0
    data = get_dataset(pre)
    n = len(data)
    for row in data:
        MSE += (beta_0 + beta_1 * row[0] - row[1])**2

    MSE = MSE / n

    return MSE

def compute_betas(pre):
    data = get_dataset(pre)
    n = len(data)
    beta_0 = 0.0
    beta_1 = 0.0
    x_bar = 0.0
    y_bar = 0.0
    for row in data:
        x_bar += row[0]
        y_bar += row[1]
    x_bar = x_bar / n
    y_bar = y_bar / n

    top_sum = 0.0
    bottom_sum = 0.0
    for row in data:
        top_sum += (row[0] - x_bar) * (row[1] - y_bar)
        bottom_sum += (row[0] - x_bar)**2

    beta_1 = top_sum / bottom_sum
    beta_0 = y_bar - beta_1 * x_bar
    MSE = regression(pre, beta_0, beta_1)

    return(beta_0, beta_1, MSE)

def predict(pre, day):
    beta_0, beta_1, MSE = compute_betas(pre)
    return beta_0 + beta_1 * day

# predict what the post stay at home order will predict 2, 4, 8 weeks will be
# there are 26 days in the post stay at home model. 0 marks 4/9, 25 marks 5/4.
# 5/25 is: 46 
# two weeks: 60
# four weeks: 74
# eight weeks: 102

print("Post model for 5/25: " + str(math.exp(predict(False, 46))))
print("two weeks after 5/25 is stay at home remains: " + str(math.exp(predict(False, 60))))
print("four weeks after 5/25 if stay at home remains: " + str(math.exp(predict(False, 74))))
print("eight weeks after 5/25 if stay at home remains: " + str(math.exp(predict(False, 102))))

# day that pre model predicts 605 cases (5/25 day predicted by post model) = 25
# two weeks: 39
# four weeks: 53
print("two weeks after 5/25 if restrictions lifted: " + str(math.exp(predict(True, 39))))
print("four weeks after 5/25 if restrictions lifted: " + str(math.exp(predict(True, 53))))
print("eight weeks after 5/25 if restrictions lifted: " + str(math.exp(predict(True, 81))))

