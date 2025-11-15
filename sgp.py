import numpy as np


def update_SGP(args, model, mat_list, threshold, feature_list=[], importance_list=[]):
    # print("Threshold: ", threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-1)
            sval_total = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            # update GPM
            feature_list.append(U[:, 0:r])
            # update importance (Eq-2)
            importance = ((args.scale_coff + 1) * S[0:r]) / (args.scale_coff * S[0:r] + max(S[0:r]))
            importance_list.append(importance)
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-4)
            act_proj = np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            r_old = feature_list[i].shape[1]  # old GPM bases
            Uc, Sc, Vhc = np.linalg.svd(act_proj, full_matrices=False)
            importance_new_on_old = np.dot(
                np.dot(feature_list[i].transpose(), Uc[:, 0:r_old]) ** 2,
                Sc[0:r_old] ** 2,
            )  ## r_old no of elm s**2 fmt
            importance_new_on_old = np.sqrt(importance_new_on_old)

            act_hat = activation - act_proj
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-5)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print("Skip Updating SGP for layer: {}".format(i + 1))
                # update importances
                importance = importance_new_on_old
                importance = ((args.scale_coff + 1) * importance) / (args.scale_coff * importance + max(importance))
                importance[0:r_old] = np.clip(importance[0:r_old] + importance_list[i][0:r_old], 0, 1)
                importance_list[i] = importance  # update importance
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            # update importance
            importance = np.hstack((importance_new_on_old, S[0:r]))
            importance = ((args.scale_coff + 1) * importance) / (args.scale_coff * importance + max(importance))
            importance[0:r_old] = np.clip(importance[0:r_old] + importance_list[i][0:r_old], 0, 1)

            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0 : Ui.shape[0]]
                importance_list[i] = importance[0 : Ui.shape[0]]
            else:
                feature_list[i] = Ui
                importance_list[i] = importance

    # print("-" * 40)
    # print("Gradient Constraints Summary")
    # print("-" * 40)
    # for i in range(len(feature_list)):
    #     print(
    #         "Layer {} : {}/{}".format(
    #             i + 1, feature_list[i].shape[1], feature_list[i].shape[0]
    #         )
    #     )
    # print("-" * 40)
    return feature_list, importance_list
