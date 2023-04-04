import numpy as np
import pandas as pd
import copy
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import csv
import torch
from torch.autograd import Variable
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import argparse, sys
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.pruning import get_noise_indices
from loader import GetLoader
from torch.utils.data import DataLoader
from modell1 import MLP
from loss import loss_coteaching
import warnings
from cleanlab.latent_estimation import (
    compute_confident_joint,
    estimate_latent,
)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.002)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.05)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--num_gradual', type = int, default = 50, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=5)
# parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')    # 多线程导致代码重复执行
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=120)

args = parser.parse_args()
batch_size = 128
numi = 14 # 输入层节点数
numh1 = 30  # 隐含层节点数
numh2 = 30
numo = 2  # 输出层节点数
# batch_size = 64
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)    # 设置当前GPU的随机生成数种子

learning_rate = args.lr

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

warnings.filterwarnings("ignore")
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)    # 每个epoch去掉的样本比例, exponent=1时适用

def get_noise(xall_new, label_new, pre_new):
    # xall_new = xall_new.numpy()
    # label_new = label_new.numpy()
    # # pre_new = pre_new.numpy()
    label_new1=copy.deepcopy(label_new)
    pre_new1 = copy.deepcopy(pre_new)
    xall_new1 = copy.deepcopy(xall_new)


    label_1 = label_new1.ravel()
    y_train2 = label_1.astype(np.int16)
    confident_joint = compute_confident_joint(
        s=y_train2,
        psx=pre_new1,  # P(s = k|x)
        thresholds=None
    )
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        s=y_train2,
        py_method='cnt',
        converge_latent_estimates=False
    )

    ordered_label_errors = get_noise_indices(
        s=y_train2,
        psx=pre_new1,
        inverse_noise_matrix=inv_noise_matrix,
        confident_joint=confident_joint,
        prune_method='prune_by_noise_rate',
    )
    # print(ordered_label_errors)

    x_mask = ~ordered_label_errors
    x_pruned = xall_new1[x_mask]
    # print(label_new_2)
    s_pruned = y_train2[x_mask]

    sample_weight = np.ones(np.shape(s_pruned))
    for k in range(2):
        sample_weight_k = 1.0 / noise_matrix[k][k]
        sample_weight[s_pruned == k] = sample_weight_k

    return x_pruned, s_pruned


def evaluate(X_test, Y_test, model1, model2):


    X_test = Variable(torch.tensor(X_test).to(torch.float32))
    Y_test = Variable(torch.tensor(Y_test).to(torch.long))

    out1 = model1(X_test)
    result1 = out1.argmax(dim=1)
    # out_1 = out1.squeeze()
    # mask1 = out_1.ge(0.5).float()
    # predict_p1 = torch.hstack((1 - out_1, out_1))
    correct1 = (result1.cpu() == Y_test).sum()

    out2 = model2(X_test)
    result2 = out2.argmax(dim=1)
    # out_2 = out2.squeeze()
    # mask2 = out_2.ge(0.5).float()
    # predict_p2 = torch.hstack((1 - out_2, out_2))
    correct2 = (result2.cpu() == Y_test).sum()

    loc_all = sum(X_test[:, 4]) + sum(X_test[:, 5])
    loc_20 = loc_all * 0.2
    r_pre1 = np.zeros((len(result1), 3))
    r_pre2 = np.zeros((len(result2), 3))
    count_t1 = 0
    count_t2 = 0

    for il in range(len(result1)):
        r_pre1[il][0] = (X_test[il][4] + X_test[il][5])
        r_pre1[il][1] = result1[il]
        r_pre1[il][2] = Y_test[il]
        if Y_test[il] == 1:
            count_t1 += 1
    idex = np.lexsort([r_pre1[:, 0], -1 * r_pre1[:, 1]])
    sorted_data = r_pre1[idex, :]
    t_loc = 0
    il = 0
    count_effort1 = 0
    while t_loc < loc_20:
        t_loc += sorted_data[il][0]
        if sorted_data[il][2] == 1:
            count_effort1 = count_effort1 + 1
        il = il + 1
    effort1 = count_effort1 / count_t1

    for i2 in range(len(result2)):
        r_pre2[i2][0] = (X_test[i2][4] + X_test[i2][5])
        r_pre2[i2][1] = result2[i2]
        r_pre2[i2][2] = Y_test[i2]
        if Y_test[i2] == 1:
            count_t2 += 1
    idex = np.lexsort([r_pre2[:, 0], -1 * r_pre2[:, 1]])
    sorted_data = r_pre2[idex, :]
    t_loc = 0
    i2 = 0
    count_effort2 = 0
    while t_loc < loc_20:
        t_loc += sorted_data[i2][0]
        if sorted_data[i2][2] == 1:
            count_effort2 = count_effort2 + 1
        i2 = i2 + 1
    effort2 = count_effort2 / count_t2

    f1_1 = metrics.f1_score(Y_test, result1, pos_label=1, average="binary")
    f1_2 = metrics.f1_score(Y_test, result2, pos_label=1, average="binary")
    acc_1 = accuracy_score(Y_test, result1)
    acc_2 = accuracy_score(Y_test, result2)
    mcc_1 = matthews_corrcoef(Y_test, result1)
    mcc_2 = matthews_corrcoef(Y_test, result2)
    prec1 = metrics.precision_score(Y_test, result1, pos_label=1)  # 精确率
    prec2 = metrics.precision_score(Y_test, result2, pos_label=1)  # 精确率
    recall1 = metrics.recall_score(Y_test, result1, pos_label=1)  # 召回率
    recall2 = metrics.recall_score(Y_test, result2, pos_label=1)  # 召回率
    fpr1, tpr1, thersholds1 = metrics.roc_curve(Y_test, result1)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    fpr2, tpr2, thersholds2 = metrics.roc_curve(Y_test, result2)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    # print(mask1[:100].ravel())
    # print(Y_test[:100].ravel())

    return acc_1, acc_2, f1_1, f1_2, mcc_1, mcc_2, prec1, prec2, recall1, recall2,  roc_auc1, roc_auc2, effort1, effort2


def train(train_loader, Y_train, epoch, model1, optimizer1, model2, optimizer2, pre_new):
    # pure_ratio_list = []
    # pure_ratio_1_list = []
    # pure_ratio_2_list = []

    correct_all1 = 0
    correct_all2 = 0
    train_total1 = 0
    train_total2 = 0

    for i, (x_data, labels, indexes) in enumerate(train_loader):
        # ind = indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break


        x_pruned = Variable(torch.tensor(x_data).to(torch.float32))
        s_pruned = Variable(torch.tensor(labels).to(torch.long))



        out_1 = model1(x_pruned)
        result1 = out_1.argmax(dim = 1)
        correct_1 = (result1.cpu() == s_pruned).sum()
        correct_all1 += correct_1
        train_total1 += x_pruned.size(0)

        out_2 = model2(x_pruned)
        result2 = out_2.argmax(dim = 1)
        correct_2 = (result2.cpu() == s_pruned).sum()
        correct_all2 += correct_2
        train_total2 += x_pruned.size(0)

        criterion = nn.BCELoss(reduction = 'none')
        s_pruned = s_pruned.ravel()
        loss_1, loss_2, = loss_coteaching(out_1, out_2, s_pruned, rate_schedule[epoch], criterion)
        # pure_ratio_1_list.append(100*pure_ratio_1)
        # pure_ratio_2_list.append(100*pure_ratio_2)

        # s_pruned = torch.unsqueeze(s_pruned, -1)

        # loss_1 = F.cross_entropy(out_1, s_pruned)
        # loss_1 = torch.sum(loss_1) / len(s_pruned)
        # loss_2 = F.cross_entropy(out_2, s_pruned)
        # loss_2 = torch.sum(loss_2) / len(s_pruned)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        prec1 = accuracy_score(s_pruned, result1)
        prec2 = accuracy_score(s_pruned, result2)
        mcc1 = matthews_corrcoef(s_pruned, result1)
        mcc2 = matthews_corrcoef(s_pruned, result2)
        f1_1 = metrics.f1_score(s_pruned, result1, pos_label=1, average="binary")
        f1_2 = metrics.f1_score(s_pruned, result2, pos_label=1, average="binary")
        # if i == 0:
        #     # print(out_1[0:30])
        #     # print(mask_1[0:100])
        #     # print(s_pruned[0:100])
        #     # print(mask_1[0:30])
        #     # print(prec1)
        #     print("mcc1: %f   mcc2: %f" % (mcc1, mcc2))
        #     print("prec1: %f   prec2: %f" % (prec1, prec2))
        #     print("f11: %f   f12: %f" % (f1_1, f1_2))


        # if (i+1) % args.print_freq == 0:
        #     print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
        #           %(epoch+1, args.n_epoch, i+1, len(Y_train)//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))
        #     # torch版本不同导致问题, 将loss_1.data[0]改成loss_1.item()

    train_acc1 = float(correct_all1) / float(train_total1)
    train_acc2 = float(correct_all2) / float(train_total2)

    return train_acc1, train_acc2

# warnings.filterwarnings('ignore')


csv_order = {0: 'activemq', 1: 'camel', 2: 'derby', 3: 'geronimo', 4: 'hbase', 5: 'hcommon', 6: 'mahout', 7: 'openjpa',
             8: 'pig', 9: 'tuscany'}
csv_num = {0: 1245, 1: 2018, 2: 1153, 3: 1856, 4: 1681, 5: 2670, 6: 420, 7: 692, 8: 467, 9: 1506}


def main():

    file_name = 'CCL.csv'
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ["datasets", "acc_1", "acc_2", "mcc_1", "mcc_2", "f1_1", "f1_2", "prec1", "prec2", "recall1", "recall2", "auc1", "auc2", "effort1", "effort2"])
    # 第一列是原始f值，第二列是运用置信学习去噪的结果，第三列是不平衡+置信学习
    for i in range(10):
        csv_string = csv_order[i]
        dataframe = pd.read_csv('dataset/' + csv_string + '.csv')
        v = dataframe.iloc[:]
        train_v = np.array(v)
        # print(train_v )
        write_all = []
        write_all.append(csv_string)

        ob = train_v[:, 0:14]
        label = train_v[:, -1]
        label = label.reshape(-1, 1)


        acc1_all = []
        acc2_all = []
        mcc1_all = []
        mcc2_all = []
        f11_all = []
        f12_all = []
        prec1_all = []
        prec2_all = []
        recall1_all = []
        recall2_all = []
        auc1_all = []
        auc2_all = []
        effort1_all = []
        effort2_all = []

        seed = [94733, 16588, 1761, 59345, 27886, 80894, 22367, 65435, 96636, 89300]
        for ix in range(10):
            sfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed[ix])

            acc1 = []
            acc2 = []
            mcc1 = []
            mcc2 = []
            f11 = []
            f12 = []
            pre1 = []
            pre2 = []
            rec1 = []
            rec2 =[]
            auc_1 = []
            auc_2 =[]
            teffort1 = []
            teffort2 = []

            for train_index, test_index in sfolder.split(ob, label):


                X_train, X_test = ob[train_index], ob[test_index]
                Y_train, Y_test = label[train_index], label[test_index]

                psx = np.zeros((len(Y_train), 2))
                psx1 = np.zeros((len(Y_train), 2))

                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed[ix])
                for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X_train, Y_train)):

                    # Select the training and holdout cross-validated sets.
                    X_train_cv, X_holdout_cv = X_train[cv_train_idx], X_train[cv_holdout_idx]
                    s_train_cv, s_holdout_cv = Y_train[cv_train_idx], Y_train[cv_holdout_idx]

                    # Fit the clf classifier to the training set and
                    # predict on the holdout set and update psx.
                    ros = RandomOverSampler(random_state=seed[ix])
                    X_resampled, y_resampled = ros.fit_resample(X_train_cv, s_train_cv)
                    log_reg = LogisticRegression(solver='liblinear')
                    log_reg.fit(X_resampled, y_resampled)
                    psx_cv = log_reg.predict_proba(X_holdout_cv)  # P(s = k|x) # [:,1]
                    psx[cv_holdout_idx] = psx_cv


                X_train, Y_train = get_noise(X_train, Y_train, psx)
                ros1 = RandomOverSampler(random_state=seed[ix])
                X_train, Y_train = ros1.fit_resample(X_train, Y_train)
                Y_train = Y_train.reshape((-1, 1))
                train_data = GetLoader(X_train, Y_train)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
                # noise_or_not = train_data.noise_or_not

                # #是否适用testloader
                # test_data = GetLoader(X_test, Y_test)
                # test_loader = DataLoader(test_data, batch_size=6, shuffle=True, drop_last=False, num_workers=2)

                mean_pure_ratio1 = 0
                mean_pure_ratio2 = 0

                epoch = 0
                train_acc1 = 0
                train_acc2 = 0

                model1 = MLP(numi, numh1, numh2, numo)
                optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

                model2 = MLP(numi, numh1, numh2, numo)
                optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

                model1.train()
                model2.train()


                for epoch in range(1, args.n_epoch):
                    adjust_learning_rate(optimizer1, epoch)
                    adjust_learning_rate(optimizer2, epoch)
                    # print(epoch)
                    train_acc1, train_acc2 = train(train_loader, Y_train, epoch, model1,
                                                   optimizer1, model2, optimizer2, psx)

                    # mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
                    # mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

                model1.eval()
                model2.eval()

                test_acc1, test_acc2, test_f11, test_f12, test_mcc1, test_mcc2, prec1, prec2, recall1, recall2, roc1, roc2, effort_1, effort_2 = evaluate(X_test, Y_test, model1, model2)
                print('%s Epoch [%d/%d] Test Accuracy on the %s test: Model1 %.4f  Model2 %.4f ,the mcc on the test: %.4f  %.4f , f1 : %.4f %.4f' % (
                   csv_string, epoch + 1, args.n_epoch, len(Y_test), test_acc1, test_acc2, test_mcc1, test_mcc2, test_f11, test_f12))

                acc1.append(test_acc1)
                acc2.append(test_acc2)
                mcc1.append(test_mcc1)
                mcc2.append(test_mcc2)
                f11.append(test_f11)
                f12.append(test_f12)
                pre1.append(prec1)
                pre2.append(prec2)
                rec1.append(recall1)
                rec2.append(recall2)
                auc_1.append(roc1)
                auc_2.append(roc2)
                teffort1.append(np.mean(effort_1))
                teffort2.append(np.mean(effort_2))

            acc1_all.append(np.mean(acc1))
            acc2_all.append(np.mean(acc2))
            mcc1_all.append(np.mean(mcc1))
            mcc2_all.append(np.mean(mcc2))
            f11_all.append(np.mean(f11))
            f12_all.append(np.mean(f12))
            prec1_all.append(np.mean(pre1))
            prec2_all.append(np.mean(pre2))
            recall1_all.append(np.mean(rec1))
            recall2_all.append(np.mean(rec2))
            auc1_all.append(np.mean(auc_1))
            auc2_all.append(np.mean(auc_2))
            # teffort1.append(np.mean(effort_1))
            # teffort2.append(np.mean(effort_2))
            effort1_all.append(np.mean(teffort1))
            effort2_all.append(np.mean(teffort2))

        write_all.append(np.mean(acc1_all))
        write_all.append(np.mean(acc2_all))
        write_all.append(np.mean(mcc1_all))
        write_all.append(np.mean(mcc2_all))
        write_all.append(np.mean(f11_all))
        write_all.append(np.mean(f12_all))
        write_all.append(np.mean(prec1_all))
        write_all.append(np.mean(prec2_all))
        write_all.append(np.mean(recall1_all))
        write_all.append(np.mean(recall2_all))
        write_all.append(np.mean(auc1_all))
        write_all.append(np.mean(auc2_all))
        write_all.append(np.mean(effort1_all))
        write_all.append(np.mean(effort2_all))
        # write_all.append(np.mean(teffort1))
        # write_all.append(np.mean(teffort2))
        print(np.mean(acc1_all), np.mean(acc2_all), np.mean(mcc1_all), np.mean(mcc2_all), np.mean(f11_all), np.mean(f12_all))

        csv_writer.writerow(write_all)
        print(csv_string + " is done~!")

if __name__ == '__main__':
    main()