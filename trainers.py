#from torchvision.models import resnet
from models.deprecated_crnn import *
from torch.autograd import Variable
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy_batch_torch, conv_weight, conv_window, PredConvolver
import logging

logger = logging.getLogger("root."+__name__)

MAX_LENGTH=60

def to_value(loss_tensor):
    return torch.sum(loss_tensor.cpu(), 0, keepdim=False).item()

class TrainerBaseline(json.JSONEncoder):
    def __init__(self, model, optimizer, config, loss_criterion):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        self.idx_to_char = self.config["idx_to_char"]
        self.train_decoder = string_utils.naive_decode
        self.decoder = config["decoder"]

        if self.config["n_warp_iterations"]:
            print("Using test warp")

    def default(self, o):
        return None

    def train(self, line_imgs, online, labels, label_lengths, gt, retain_graph=False, step=0):
        self.model.train()

        pred_tup = self.model(line_imgs, online)
        pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        # Calculate HWR loss
        preds_size = Variable(torch.IntTensor([pred_logits.size(0)] * pred_logits.size(1))) # <- what? isn't this square? why are we tiling the figsize?

        output_batch = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        pred_strs = list(self.decoder.decode_training(output_batch))

        # Get losses
        logger.debug("Calculating CTC Loss: {}".format(step))
        loss_recognizer = self.loss_criterion(pred_logits, labels, preds_size, label_lengths)

        # Backprop
        logger.debug("Backpropping: {}".format(step))
        self.optimizer.zero_grad()
        loss_recognizer.backward(retain_graph=retain_graph)
        self.optimizer.step()

        loss = torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item()

        # Error Rate
        self.config["stats"]["HWR Training Loss"].accumulate(loss, 1) # Might need to be divided by batch figsize?
        logger.debug("Calculating Error Rate: {}".format(step))
        err, weight = calculate_cer(pred_strs, gt)

        logger.debug("Accumulating stats")
        self.config["stats"]["Training Error Rate"].accumulate(err, weight)

        return loss, err, pred_strs

    def test(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True, with_iterations=False):
        if with_iterations:
            self.config.logger.debug("Running test with iterations")
            return self.test_warp(line_imgs, online, gt, force_training, nudger, validation=validation)
        else:
            self.config.logger.debug("Running normal test")
            return self.test_normal(line_imgs, online, gt, force_training, nudger, validation=validation)

    def test_normal(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True):
        """

        Args:
            line_imgs:
            online:
            gt:
            force_training: Run test in .train() as opposed to .eval() mode

        Returns:

        """

        if force_training:
            self.model.train()
        else:
            self.model.eval(device=self.config.device)

        pred_tup = self.model(line_imgs, online)
        pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]

        output_batch = pred_logits.permute(1, 0, 2)
        pred_strs = list(self.decoder.decode_test(output_batch))

        # Error Rate
        if nudger:
            return rnn_input
        else:
            err, weight = calculate_cer(pred_strs, gt)
            self.update_test_cer(validation, err, weight)
            loss = -1 # not calculating test loss here
            return loss, err, pred_strs

    def update_test_cer(self, validation, err, weight, prefix=""):
        if validation:
            self.config.logger.debug("Updating validation!")
            stat = self.config["designated_validation_cer"]
            self.config["stats"][f"{prefix}{stat}"].accumulate(err, weight)
        else:
            self.config.logger.debug("Updating test!")
            stat = self.config["designated_test_cer"]
            self.config["stats"][f"{prefix}{stat}"].accumulate(err, weight)
            #print(self.config["designated_test_cer"], self.config["stats"][f"{prefix}{stat}"])

    def test_warp(self, line_imgs, online, gt, force_training=False, nudger=False, validation=True):
        if force_training:
            self.model.train()
        else:
            self.model.eval(device=self.config.device)

        #use_lm = config['testing_language_model']
        #n_warp_iterations = config['n_warp_iterations']

        compiled_preds = []
        # Loop through identical images
        # batch, repetitions, c/h/w
        for n in range(0, line_imgs.shape[1]):
            imgs = line_imgs[:,n,:,:,:]
            pred_tup = self.model(imgs, online)
            pred_logits, rnn_input, *_ = pred_tup[0].cpu(), pred_tup[1], pred_tup[2:]
            output_batch = pred_logits.permute(1, 0, 2)
            pred_strs = list(self.decoder.decode_test(output_batch))
            compiled_preds.append(pred_strs) # reps, batch

        compiled_preds = np.array(compiled_preds).transpose((1,0)) # batch, reps

        # Loop through batch items
        best_preds = []
        for b in range(0, compiled_preds.shape[0]):
            preds, counts = np.unique(compiled_preds[b], return_counts=True)
            best_pred = preds[np.argmax(counts)]
            best_preds.append(best_pred)

        # Error Rate
        if nudger:
            return rnn_input
        else:
            err, weight = calculate_cer(best_preds, gt)
            self.update_test_cer(validation, err, weight)
            loss = -1 # not calculating test loss here
            return loss, err, pred_strs


class Trainer:
    def __init__(self, model, optimizer, config, loss_criterion=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        if config is None:
            self.logger = utils.setup_logging()
        else:
            self.logger = config.logger

    @staticmethod
    def truncate(preds, label_lengths):
        """ Take in rectangular GT tensor, return as list where each element in batch has been truncated

        Args:
            preds:
            label_lengths:

        Returns:

        """

        preds = [preds[i][:label_lengths[i], :] for i in range(0, len(label_lengths))]
        return preds

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    def eval(self, **kwargs):
        raise NotImplemented

    def train(self, **kwargs):
        raise NotImplemented


class TrainerStrokeRecovery:
    def __init__(self, model, optimizer, config, loss_criterion=None):
        #super().__init__(model, optimizer, config, loss_criterion)
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_criterion = loss_criterion
        if config is None:
            self.logger = utils.setup_logging()
        else:
            self.logger = config.logger
        self.opts = None
        self.relative = None
        self.update_relative(config.pred_opts)
        if config.convolve_func == "cumsum":
            self.convolve = None # use relativefy
        else:
            self.convolve = PredConvolver(config.convolve_func, kernel_length=config.cumsum_window_size).convolve

    def default(self, o):
        return None

    def update_relative(self, pred_opts):
        self.relative = [i for i,x in enumerate(pred_opts) if x=="cumsum"]
        return self.relative

    @staticmethod
    def truncate(preds, label_lengths):
        """ Return a list

        Args:
            preds:
            label_lengths:

        Returns:

        """

        preds = [preds[i][:label_lengths[i], :] for i in range(0, len(label_lengths))]
        #targs = [targs[i][:label_lengths[i], :] for i in range(0, len(label_lengths))]
        return preds

    def train(self, item, train=True, **kwargs):
        """ Item is the whole thing from the dataloader

        Args:
            loss_fn:
            item:
            train: train/update the model
            **kwargs:

        Returns:

        """
        line_imgs = item["line_imgs"].to(self.config.device)
        label_lengths = item["label_lengths"]
        gt = item["gt_list"]
        suffix = "_train" if train else "_test"

        if train:
            self.model.train()
            self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)
            #print(self.config.stats[])

        preds = self.eval(line_imgs, self.model, label_lengths=label_lengths, relative=self.relative,
                          device=self.config.device, gt=item["gt"], train=train, convolve=self.convolve)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

        loss_tensor, loss = self.loss_criterion.main_loss(preds, gt, label_lengths, suffix)

        # Update all other stats
        self.update_stats(item, preds, train=train)

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()
        return loss, preds, None

    def test(self, item, **kwargs):
        self.model.eval()
        return self.train(item, train=False, **kwargs)

    @staticmethod
    def eval(line_imgs, model, label_lengths=None, relative=None, device="cuda", gt=None, train=False, convolve=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs).cpu()
        preds = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab

        ## Make absolute preds from relative preds - must be done before truncation
        if relative:
            if not train or convolve is None:
                preds = relativefy_batch_torch(preds, reverse=True, indices=relative)  # assume they were in relative positions, convert to absolute
            else:
                preds = convolve(pred_rel=preds, indices=relative, gt=gt)

        ## Shorten - label lengths currently = width of image after CNN
        if not label_lengths is None:
            preds = TrainerStrokeRecovery.truncate(preds, label_lengths) # Convert square torch object to a list, removing predictions related to padding
        return preds

    def update_stats(self, item, preds, train=True):
        suffix = "_train" if train else "_test"

        ## If not using L1 loss, report the stat anyway
        # if "l1" not in self.loss_criterion.loss_names:
        #     # Just a generic L1 loss for x,y coords
        #     l1_loss = to_value(self.config.L1.lossfun(preds, item["gt_list"], item["label_lengths"])) # don't divide by batch figsize
        #     self.config.stats["l1"+suffix].accumulate(l1_loss)

        # Don't do the nearest neighbor search by default
        if (self.config.training_nn_loss and train) or (self.config.test_nn_loss and not train) :
            self.config.stats["nn"+suffix].accumulate(self.loss_criterion.calculate_nn_distance(item, preds))


class TrainerStartPoints(Trainer):
    def __init__(self, model, optimizer, config, loss_criterion=None):
        super().__init__(model, optimizer, config, loss_criterion)
        self.opts = None
        self.relative = None

    def train(self, item, train=True, **kwargs):
        """ Item is the whole thing from the dataloader

        Args:
            loss_fn:
            item:
            train: train/update the model
            **kwargs:

        Returns:

        """
        line_imgs = item["line_imgs"].to(self.config.device)
        label_lengths = item["label_lengths"]
        gt = item["start_points"]
        suffix = "_train" if train else "_test"

        ## Filter GTs to just the start points and the EOS point; the EOS point will be the finish point of the last stroke
        if train:
            self.model.train()
            self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)

        preds = self.eval(line_imgs, self.model, label_lengths=label_lengths,
                          device=self.config.device, train=train)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

        # Shorten pred to be the length of the ground truth
        pred_list = []
        for i, pred in enumerate(preds):
            pred_list.append(pred[:len(gt[i])])

        loss_tensor, loss = self.loss_criterion.main_loss(pred_list, gt, label_lengths, suffix)

        if train:
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()
        return loss, preds, None

    @staticmethod
    def eval(line_imgs, model, label_lengths=None, device="cuda", train=False, convolve=None):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs).cpu()
        preds = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        return preds
