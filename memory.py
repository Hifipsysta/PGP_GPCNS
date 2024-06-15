import torch
import numpy as np
from sklearn.decomposition import PCA


def get_prefix_matrix(data_loader, model, device, prompt_id=None):
    model.eval()
    count = 1
    representation_g = {}
    representation_e = {}

    # gradient_g = {}
    # gradient_e = {}

    ###############################
    prefix_grad_g, prefix_grad_e = [], []
    # if epoch == args.epochs - 1:
    for k, (m, params) in enumerate(model.module.named_parameters()):
        if m == "g_prompt":
            # if params.grad.data.size(0) == 24:
            
            for jj in range(params.grad.data.size(0)):
                # print('g_prompt:  params.grad.data.shape', params.grad.data[jj][0].shape)
                prefix_grad_g.append(params.grad.data[jj][0])

        if m == "e_prompt.prompt":
            # if params.grad.data.size(0) == 24:
            
            for jj in range(params.grad.data.size(0)):
                # print('e_prompt.prompt:  params.grad.data.shape', params.grad.data[jj][0][prompt_id].shape)
                prefix_grad_e.append(params.grad.data[jj][0][prompt_id])

    ##################################

    ####################
    # for layer in model.module.g_prefix_gradient:
    #     if layer not in gradient_g:
    #         gradient_g[layer] = {"key": []}
    #         print('model.module.g_prefix_gradient[layer]["key"]', model.module.g_prefix_gradient[layer]["key"])
    #     gradient_g[layer]["key"].append(model.module.g_prefix_gradient[layer]["key"])

    # for layer in model.module.e_prefix_gradient:
    #     if layer not in gradient_e:
    #         gradient_e[layer] = {"key": []}
    #         print('model.module.e_prefix_gradient[layer]["key"]', model.module.e_prefix_gradient[layer]["key"])
    #     gradient_e[layer]["key"].append(model.module.e_prefix_gradient[layer]["key"])
    
    #####################


    with torch.no_grad():
        for input, target in data_loader:
            # print('input', input.shape)
            input = input.to(device, non_blocking=True)
            # prefix = torch.randn(1, 2, 5, 12, 64)
            _ = model(input)
            del _

            for layer in model.module.g_prefix_feature:
                if layer not in representation_g:
                    representation_g[layer] = {"key": []}
                representation_g[layer]["key"].append(model.module.g_prefix_feature[layer]["key"])
            for layer in model.module.e_prefix_feature:
                if layer not in representation_e:
                    representation_e[layer] = {"key": []}
                representation_e[layer]["key"].append(model.module.e_prefix_feature[layer]["key"])
            count += 1

            if count > 768:
                for layer in representation_g:
                    for item in representation_g[layer]:
                        # print('representation_g[layer][item]', representation_g[layer][item])

                        representation_g[layer][item] = torch.cat(representation_g[layer][item])
                        representation_g[layer][item] = representation_g[layer][item].detach().cpu().numpy()
                        representation_g[layer][item] = representation_g[layer][item].reshape(representation_g[layer][item].shape[0], -1)
                        rep = representation_g[layer][item]
                        # print('Layer: ', layer, 'Size of representation_g: ', rep.shape)
                        pca = PCA(n_components=50)
                        pca = pca.fit(rep)
                        rep = pca.transform(rep)
                        representation_g[layer][item] = rep

                for layer in representation_e:
                    for item in representation_e[layer]:
                        # print('representation_e[layer][item]', representation_e[layer][item])

                        representation_e[layer][item] = torch.cat(representation_e[layer][item])
                        representation_e[layer][item] = representation_e[layer][item].detach().cpu().numpy()
                        representation_e[layer][item] = representation_e[layer][item].reshape(representation_e[layer][item].shape[0], -1)
                        rep = representation_e[layer][item]
                        # print('Layer: ', layer, 'Size of representation_e: ', rep.shape)
                        pca = PCA(n_components=50)
                        pca = pca.fit(rep)
                        rep = pca.transform(rep)
                        representation_e[layer][item] = rep

                break
    torch.cuda.empty_cache()

    return representation_g, representation_e, prefix_grad_g, prefix_grad_e


def update_memory_prefix(represent, threshold, features=None, prefix_grad=None, grad_stack_list=[], space_list=[], feature_list=[]):
    ii = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for layer in represent:
        for item in represent[layer]:
            representation = represent[layer][item]
            # print('representation', len(representation))
            
            representation = np.matmul(representation, representation.T)
            try:
                feature = features[layer][item]
            except:
                feature = None


            if feature is None:
                if layer not in features:
                    features[layer] = {}


                Gradient_matrix = torch.mm(prefix_grad[ii].reshape(-1, 768).transpose(0,1), prefix_grad[ii].reshape(-1, 768))

                print('Gradient Matrix', Gradient_matrix.shape)
                grad_stack_list.append(Gradient_matrix)

                U_matrix, Sigma, Vh_matrix = torch.linalg.svd(grad_stack_list[ii], full_matrices=False)
                V_matrix = Vh_matrix.transpose(0,1)
                
                Sigma_total = torch.sum(Sigma)
                Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]) for idx in range(len(Sigma))]
                Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= 0.98*Sigma_total
                Sigma_slim = Sigma[Sigma_select]
                rank_subspace = torch.count_nonzero(Sigma_slim)
                print('Rank of Matrix：', rank_subspace)

                space_list.append(V_matrix[:, :rank_subspace])
                print('space_list[ii]', space_list[ii].shape)

                



                U, S, Vh = np.linalg.svd(representation, full_matrices=False)
                print('Layer', layer, 'representation', representation.shape, 'prefix_grad', len(prefix_grad))
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)


                feature_list.append(torch.tensor(U[:,0:r]).float().to(device))

                U_matrix, Sigma, Vh_matrix = torch.linalg.svd(torch.hstack([space_list[ii], feature_list[ii]]), full_matrices=False)

                Sigma_total = torch.sum(Sigma)
                Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
                Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= 0.98*Sigma_total
                Sigma_slim = Sigma[Sigma_select]
                rank_subspace = torch.count_nonzero(Sigma_slim)


                space_list[ii] = U_matrix[:, :rank_subspace]
                print('Revised space_list[ii]', space_list[ii].shape)

                feature = U_matrix[:, :rank_subspace]


                ii += 1



            else:
                Gradient_matrix = torch.mm(prefix_grad[ii].reshape(-1, 768).transpose(0,1), prefix_grad[ii].reshape(-1, 768))
                print('Gradient Matrix', Gradient_matrix.shape)
                grad_stack_list[ii] = torch.vstack([grad_stack_list[ii], Gradient_matrix])   
                print('Matrix waiting to be decomposed', grad_stack_list[ii].shape)

                U_matrix, Sigma, Vh_matrix = torch.linalg.svd(grad_stack_list[ii], full_matrices=False) 
                V_matrix = Vh_matrix.transpose(0,1)

                Sigma_total = torch.sum(Sigma)
                Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
                Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= 0.98*Sigma_total
                Sigma_slim = Sigma[Sigma_select]
                rank_subspace = torch.count_nonzero(Sigma_slim)
                print('Rank of Matrix：', rank_subspace)

                space_list[ii] = V_matrix[:, :rank_subspace]
                print('space_list[ii]', space_list[ii].shape)



                U, S, Vh = np.linalg.svd(representation, full_matrices=False)
                print('Layer', layer, 'representation', representation.shape, 'prefix_grad', len(prefix_grad))
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)
                # feature = U[:, 0:r]


                feature_list[ii] = torch.hstack([feature_list[ii], torch.tensor(U[:,0:r]).float().to(device)])

                U_matrix, Sigma, Vh_matrix = torch.linalg.svd(torch.hstack([space_list[ii], feature_list[ii]]), full_matrices=False)


                Sigma_total = torch.sum(Sigma)
                Sigma_accumul_list = [torch.sum(Sigma[0:idx+1]).item() for idx in range(len(Sigma))]
                Sigma_select = torch.tensor(Sigma_accumul_list).to(device) <= 0.98*Sigma_total
                Sigma_slim = Sigma[Sigma_select]
                rank_subspace = torch.count_nonzero(Sigma_slim)

                space_list[ii] = U_matrix[:, :rank_subspace]
                print('Revised space_list[ii]', space_list[ii].shape)

                feature = U_matrix[:, :rank_subspace]

                # U1, S1, Vh1 = np.linalg.svd(representation, full_matrices=False)
                # sval_total = (S1 ** 2).sum()
                # # Projected Representation
                # act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
                # U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # print('Layer', layer, 'act_hat', act_hat.shape, 'prefix_grad', len(prefix_grad))
                # # criteria
                # sval_hat = (S ** 2).sum()
                # sval_ratio = (S ** 2) / sval_total
                # accumulated_sval = (sval_total - sval_hat) / sval_total
                # r = 0
                # for ii in range(sval_ratio.shape[0]):
                #     if accumulated_sval < threshold:
                #         accumulated_sval += sval_ratio[ii]
                #         r += 1
                #     else:
                #         break
                # if r == 0:
                #     feature = feature
                # # update GPM
                # U = np.hstack((feature, U[:, 0:r]))
                # if U.shape[1] > U.shape[0]:
                #     feature = U[:, 0:U.shape[0]]
                # else:
                #     feature = U

                ii += 1

            print('-'*40)
            print('Gradient Constraints Summary', feature.shape)
            print('-'*40)
            features[layer][item] = feature

    return features, grad_stack_list, space_list, feature_list
