from sklearn.metrics import average_precision_score
import torch

y_true = torch.tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])

y_score = torch.tensor(([[ 0.3237, -4.3488,  2.2320,  1.4222,  1.0371, -0.4771,  0.8291,  1.6571,
          0.3955, -3.0881],
        [ 0.9164, -4.5222,  2.3210,  1.2271,  1.0176, -0.6508,  0.9205,  1.2561,
          0.2573, -2.7375],
        [-0.8520, -5.0340,  2.2008,  2.1995,  2.0658, -0.4351,  1.0123,  2.8260,
          0.6814, -4.7314]]))


# print(len(y_true))
# print(len(y_score))
# ap = []
# for i in range(3):
#     ap.append(average_precision_score(y_true[i], y_score[i]))
# print(sum(ap)/len(ap))
for i in range(len(y_score)):
    for j in range(len(y_score[i])):
        if y_score[i][j] == 1:
            class_pred = j
            print(class_pred)
    if class_pred not in avgprec:
        avgprec[class_pred] = list()
        avgprec[class_pred].append(average_precision_score(labels[i], cpuout[i]))
    else:
        avgprec[class_pred].append(average_precision_score(labels[i], cpuout[i]))