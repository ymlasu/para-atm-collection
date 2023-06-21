import argparse
import time, pickle
import dgl
import torch
import torch.nn.functional as F
from dataset import EllipticDataset
from model import EvolveGCNO, EvolveGCNH
from utils import Measure
from sklearn.metrics import f1_score

def train(args, device):
    elliptic_dataset = EllipticDataset(raw_dir=args.raw_dir,
                                       processed_dir=args.processed_dir,
                                       self_loop=True,
                                       reverse_edge=True)

    g, node_mask_by_time = elliptic_dataset.process()
    num_classes = elliptic_dataset.num_classes

    cached_subgraph = []
    cached_labeled_node_mask = []
    for i in range(len(node_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        cached_subgraph.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata['label'] >= 0
        cached_labeled_node_mask.append(valid_node_mask)

    if args.model == 'EvolveGCN-O':
        model = EvolveGCNO(in_feats=int(g.ndata['feat'].shape[1]),
                           n_hidden=args.n_hidden,
                           num_layers=args.n_layers)
    elif args.model == 'EvolveGCN-H':
        model = EvolveGCNH(in_feats=int(g.ndata['feat'].shape[1]),
                           num_layers=args.n_layers)
    else:
        return NotImplementedError('Unsupported model {}'.format(args.model))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # split train, valid, test dataset
    split = [0.4, 0.3, 0.3]
    train_max_index = args.n_hist_steps * \
        int(split[0]*len(cached_subgraph)//args.n_hist_steps)
    valid_max_index = train_max_index + args.n_hist_steps * \
        int(split[1]*len(cached_subgraph)//args.n_hist_steps)
    test_max_index = valid_max_index + args.n_hist_steps * \
        int(split[2]*len(cached_subgraph)//args.n_hist_steps)

    ########################################
    # debug only
    # REMOVE START
    # split = [0.7, 0, 0]
    # train_max_index = args.n_hist_steps * \
    #     int(split[0]*len(cached_subgraph)//args.n_hist_steps)
    # valid_max_index = train_max_index
    # test_max_index = valid_max_index
    # REMOVE END
    ########################################

    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(',')]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)

    train_measure = Measure(num_classes=num_classes,
                            target_class=args.eval_class_id)
    valid_measure = Measure(num_classes=num_classes,
                            target_class=args.eval_class_id)
    test_measure = Measure(num_classes=num_classes,
                           target_class=args.eval_class_id)

    test_res_f1 = 0
    for epoch in range(args.num_epochs):

        print(f'Epoch {epoch}  >>>>>>>>>>>>>')

        model.train()
        pred_val, label_val = [], [] # save for conformal prediction

        for i in range(time_window_size, train_max_index+1):
            
            g_list = cached_subgraph[i-time_window_size: i] 
            predictions = model(g_list)

            # get predictions which has label
            predictions = predictions[cached_labeled_node_mask[i-1][0].unsqueeze(dim=0)]

            labels = cached_subgraph[i-1].ndata['label'][cached_labeled_node_mask[i-1]].long()[0].unsqueeze(dim=0)

            # import pdb;pdb.set_trace()
            loss = F.cross_entropy(predictions, labels,
                                   weight=loss_class_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_measure.append_measures(predictions, labels)

            pred_val.append(predictions)
            label_val.append(labels)

        # save best results
        best_results = {'pred': pred_val,
                        'label': label_val
                        }
        with open(f'{args.raw_dir}/best_results_epoch_{epoch}.pkl', 'wb') as handle:
            pickle.dump(best_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # get each epoch measures during training.
        # cl_precision, cl_recall, cl_f1 = train_measure.get_total_measure()
        micro_f1, macro_f1 = train_measure.get_f1_measure()
        train_measure.update_best_f1(micro_f1, epoch)

        # reset measures for next epoch
        train_measure.reset_info()

        print("Train | microF1:{:.4f} | macroF1: {:.4f}| Trainloss:{:.8f}"
              .format(micro_f1, macro_f1, loss))

        # eval on validation set
        model.eval()

        for i in range(train_max_index + 1, valid_max_index + 1):

            g_list = cached_subgraph[i-time_window_size:i]
            predictions = model(g_list)
            # get node predictions which has label
            predictions = predictions[cached_labeled_node_mask[i-1][0].unsqueeze(dim=0)]
            labels = cached_subgraph[i-1].ndata['label'][cached_labeled_node_mask[i-1]].long()[0].unsqueeze(dim=0)

            # print(labels)
            loss = F.cross_entropy(predictions, labels,
                                   weight=loss_class_weight)

            valid_measure.append_measures(predictions, labels)

        # get each epoch measure during eval.
        # cl_precision, cl_recall, cl_f1 = valid_measure.get_total_measure()
        micro_f1, macro_f1 = valid_measure.get_f1_measure()

        valid_measure.update_best_f1(micro_f1, epoch)

        # reset measures for next epoch
        valid_measure.reset_info()

        print("Valid | microF1:{:.4f} | macroF1: {:.4f} | Validloss:{:.8f}"
              .format(micro_f1, macro_f1, loss))

        # early stop
        if epoch - valid_measure.target_best_f1_epoch >= args.patience:
            print("Best eval Epoch {}, Cur Epoch {}".format(
                valid_measure.target_best_f1_epoch, epoch))
            break

        # if cur valid f1 score is best, do test
        if epoch == valid_measure.target_best_f1_epoch:
            print("###################Epoch {} Test###################".format(epoch))

            for i in range(train_max_index + 1, test_max_index + 1):

                g_list = cached_subgraph[i-time_window_size:i]
                predictions = model(g_list)

                # get predictions which has label
                predictions = predictions[cached_labeled_node_mask[i-1][0].unsqueeze(dim=0)]
                labels = cached_subgraph[i-1].ndata['label'][cached_labeled_node_mask[i-1]].long()[0].unsqueeze(dim=0)

                test_measure.append_measures(predictions, labels)

            # # we get each subgraph measure when testing to match fig 4 in EvolveGCN paper.
            # cl_precisions, cl_recalls, cl_f1s = test_measure.get_each_timestamp_measure()
            # for index, (sub_p, sub_r, sub_f1) in enumerate(zip(cl_precisions, cl_recalls, cl_f1s)):
            #     print("  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
            #           .format(train_max_index + index + 2, sub_p, sub_r, sub_f1))

            # get each epoch measure during test.
            micro_f1, macro_f1 = test_measure.get_f1_measure()
            test_measure.update_best_f1(micro_f1, epoch)

            # reset measures for next test
            test_measure.reset_info()

            test_res_f1 = micro_f1

            print("Test | Epoch {} | microF1:{:.4f} | macroF1: {:.4f}"
                  .format(epoch, micro_f1, macro_f1))

    print("Best test f1 is {}, in Epoch {}"
          .format(test_measure.target_best_f1, test_measure.target_best_f1_epoch))
    if test_measure.target_best_f1_epoch != valid_measure.target_best_f1_epoch:
        print("The Epoch get best Valid measure not get the best Test measure, "
              "please checkout the test result in Epoch {}, which f1 is {}"
              .format(valid_measure.target_best_f1_epoch, test_res_f1))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument('--model', type=str, default='EvolveGCN-O',
                           help='We can choose EvolveGCN-O or EvolveGCN-H,'
                                'but the EvolveGCN-H performance on our dataset is not good.')
    argparser.add_argument('--raw-dir', type=str,
                           default='/media/ypang6/paralab/Research/ATC_WORKLOAD/DynamicGraph/workload_data/',
                           help="Dir after unzip downloaded dataset, which contains 3 csv files.")
    argparser.add_argument('--processed-dir', type=str,
                           default='/media/ypang6/paralab/Research/ATC_WORKLOAD/DynamicGraph/workload_data/processed/',
                           help="Dir to store processed raw data.")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training.")
    argparser.add_argument('--num-epochs', type=int, default=200)
    argparser.add_argument('--n-hidden', type=int, default=64)
    argparser.add_argument('--n-layers', type=int, default=4)
    argparser.add_argument('--n-hist-steps', type=int, default=36,
                           help="If it is set to 36, it means in the first batch,"
                                "we use historical data of 0-35 to predict the workload at time 35.")
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--loss-class-weight', type=str, default='0.3, 0.1, 0.1, 0.05, 0.05, 0.1, 0.3',
                           help='Weight for loss function.')
    argparser.add_argument('--patience', type=int, default=20,
                           help="Patience for early stopping.")
    argparser.add_argument('--eval-class-id', type=int, default=1, ## This argument is void since we are evaluating the microf1 across all classes
                           help="Class type to eval. In our work, we are interested in the averaged F1 score")

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
