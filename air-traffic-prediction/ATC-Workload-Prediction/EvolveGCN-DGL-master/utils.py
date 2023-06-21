def calculate_measure(tp, fn, fp):
    # avoid nan
    if tp == 0:
        return 0, 0, 0

    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    if (p + r) > 0:
        f1 = 2.0 * (p * r) / (p + r)
    else:
        f1 = 0
    return p, r, f1


class Measure(object):
    def __init__(self, num_classes, target_class=None):
        """

        Args:
            num_classes: number of classes.
            target_class: target class we focus on, used to print info and do early stopping.
        """
        self.num_classes = num_classes
        self.target_class = target_class
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.target_best_f1 = 0.0
        self.target_best_f1_epoch = 0
        self.reset_info()

    def reset_info(self):
        """
            reset info after each epoch.
        """
        self.true_positives = {cur_class: []
                               for cur_class in range(self.num_classes)}
        self.false_positives = {cur_class: []
                                for cur_class in range(self.num_classes)}
        self.false_negatives = {cur_class: []
                                for cur_class in range(self.num_classes)}

    def append_measures(self, predictions, labels):
        predicted_classes = predictions.argmax(dim=1)

        for cl in range(self.num_classes):

            cl_indices = (labels == cl)
            pos = (predicted_classes == cl)
            hits = (predicted_classes[cl_indices] == labels[cl_indices])
            # import pdb; pdb.set_trace()
            tp = hits.sum()
            fn = hits.size(0) - tp
            fp = pos.sum() - tp

            self.true_positives[cl].append(tp.cpu())
            self.false_negatives[cl].append(fn.cpu())
            self.false_positives[cl].append(fp.cpu())
        # breakpoint()

    def append_conformal_measure(self, prediction_set, labels):
        return

    def get_each_timestamp_measure(self):
        precisions = []
        recalls = []
        f1s = []
        for i in range(len(self.true_positives[self.target_class])):
            tp = self.true_positives[self.target_class][i]
            fn = self.false_negatives[self.target_class][i]
            fp = self.false_positives[self.target_class][i]

            p, r, f1 = calculate_measure(tp, fn, fp)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        return precisions, recalls, f1s

    def get_total_measure(self):
        tp = sum(self.true_positives[self.target_class])
        fn = sum(self.false_negatives[self.target_class])
        fp = sum(self.false_positives[self.target_class])

        p, r, f1 = calculate_measure(tp, fn, fp)
        return p, r, f1

    def update_best_f1(self, cur_f1, cur_epoch):
        if cur_f1 > self.target_best_f1:
            self.target_best_f1 = cur_f1
            self.target_best_f1_epoch = cur_epoch

    def get_f1_measure(self):
        _, _, microf1 = self.get_MICROaverage_measure()
        _, _, macrof1 = self.get_MACROaverage_measure()
        return microf1, macrof1

    def get_MICROaverage_measure(self):

        tp, fn, fp = 0, 0, 0
        for i in range(self.num_classes):
            tp += sum(self.true_positives[i])
            fn += sum(self.false_negatives[i])
            fp += sum(self.false_positives[i])

        p, r, f1 = calculate_measure(tp, fn, fp)
        return p, r, f1

    def get_MACROaverage_measure(self):

        # pa, ra, f1a = [0]*self.num_classes, [0] * \
        #     self.num_classes, [0]*self.num_classes
        # for i in range(self.num_classes):

        #     p, r, f1 = calculate_measure(sum(self.true_positives[i]),
        #                                  sum(self.false_negatives[i]),
        #                                  sum(self.false_positives[i]))
        #     pa[i], ra[i], f1a[i] = p, r, f1

        # # remove zero classes
        # pa, ra, f1a = self.remove_zeros(
        #     pa), self.remove_zeros(ra), self.remove_zeros(f1a)

        # # check if all empty
        # if len(pa) or len(ra) or len(f1a) == 0:
        #     return 0, 0, 0

        # # check size equal
        # assert len(pa) == len(f1a)
        # assert len(pa) == len(ra)

        # return sum(pa)/len(pa), sum(ra)/len(ra), sum(f1a)/len(f1a)


        a = self.num_classes
        ff = 0
        pp = 0
        rr = 0
        for i in range(self.num_classes):
            if sum(self.true_positives[i]) == 0 and sum(self.false_negatives[i]) ==0:
                a -= 1
                continue

            p, r, f1 = calculate_measure(sum(self.true_positives[i]),
                                         sum(self.false_negatives[i]),
                                         sum(self.false_positives[i]))
            ff += f1
            pp += p
            rr += r
        return pp/a, rr/a, ff/a

    @staticmethod
    def remove_zeros(ls):
        return list(filter(lambda num: num != 0, ls))
