#from torchvision.models import resnet
from models.CRCR import CRCR
from models.deprecated_crnn import *
from torch.autograd import Variable
from models.basic import BidirectionalRNN, CNN
from models.CoordConv import CoordConv
from hwr_utils import utils
from hwr_utils.stroke_recovery import relativefy, relativefy_batch, relativefy_batch_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH=60

class basic_CRNN(nn.Module):
    """ CRNN with writer classifier
    """
    def __init__(self, cnnOutSize, nc, alphabet_size, rnn_hidden_dim, rnn_layers=2, leakyRelu=False,
                 recognizer_dropout=.5, rnn_input_dimension=1024, rnn_constructor=nn.LSTM, cnn_type="default", coord_conv=False):
        super().__init__()
        self.softmax = nn.LogSoftmax()
        self.dropout = recognizer_dropout

        first_conv_op = CoordConv if coord_conv else nn.Conv2d

        if cnn_type in ["default", "intermediates"] or "resnet" in cnn_type:
            self.cnn = CNN(cnnOutSize, nc, leakyRelu=leakyRelu, cnn_type=cnn_type, first_conv_op=first_conv_op)
        elif cnn_type=="crcr":
            self.cnn = CRCR(cnnOutSize, nc, leakyRelu=leakyRelu, type=cnn_type)
        else:
            raise Exception("Invalid CNN specified")
        self.rnn = BidirectionalRNN(rnn_input_dimension, rnn_hidden_dim, alphabet_size, dropout=recognizer_dropout, num_layers=rnn_layers, rnn_constructor=rnn_constructor)

    def my_eval(self):
        self.rnn.rnn.dropout = 0

    def my_train(self):
        self.rnn.rnn.dropout = self.dropout

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.my_eval()

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.my_train()

    def forward(self, input, online=None, classifier_output=None):
        """

        Args:
            input:
            online:
            classifier_output:

        Returns:
            tuple: normal prediction, refined prediction, normal CNN encoding, nudged CNN encoding

        """
        conv = self.cnn(input)
        rnn_input = conv # [width/time, batch, feature_maps]

        if online is not None:
            rnn_input = torch.cat([rnn_input, online.expand(conv.shape[0], -1, -1)], dim=2)
        recognizer_output = self.rnn(rnn_input)
        return recognizer_output, rnn_input

def create_CRNN(config):
    check_inputs(config)
    # For apples-to-apples comparison, CNN outsize is OUT_SIZE + EMBEDDING_SIZE
    crnn = basic_CRNN(cnnOutSize=config['cnn_out_size'], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                recognizer_dropout=config["recognizer_dropout"], rnn_input_dimension=config["rnn_input_dimension"], rnn_layers=config["rnn_layers"],
                      rnn_constructor=config["rnn_constructor"], cnn_type=config["cnn"])
    return crnn

def check_inputs(config):
    if not config["style_encoder"] or config["style_encoder"] in ["2StageNudger", "2Stage"]:
        use_writer_classifier = False
        config["embedding_size"] = 0
        config["num_of_writers"] = 0
        config['writer_rnn_output_size'] = 0
        config["embedding_size"] = 0
        config["writer_dropout"] = 0
        config["mlp_layers"] = []

    # Setup RNN input dimension
    config["rnn_input_dimension"] = config["cnn_out_size"] + config["embedding_size"]

    if config["online_augmentation"] and config["online_flag"]:
        config["rnn_input_dimension"] += 1

    if config["rnn_type"].lower() == "gru":
        config["rnn_constructor"]=nn.GRU
    elif config["rnn_type"].lower() == "lstm" or True:
        config["rnn_constructor"]=nn.LSTM
    return config

def create_CRNNClassifier(config, use_writer_classifier=True):
    # Don't use writer classifier
    check_inputs(config)
    crnn = CRNN_with_writer_classifier(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], nh=config["rnn_dimension"],
                                       number_of_writers=config["num_of_writers"], writer_rnn_output_size=config['writer_rnn_output_size'],
                                       embedding_size=config["embedding_size"],
                                       writer_dropout=config["writer_dropout"], recognizer_dropout=config["recognizer_dropout"],
                                       writer_rnn_dimension=config["writer_rnn_dimension"],
                                       mlp_layers=config["mlp_layers"], detach_embedding=config["detach_embedding"],
                                       online_augmentation=config["online_augmentation"], use_writer_classifier=use_writer_classifier, rnn_constructor=config["rnn_constructor"])
    return crnn

def create_2Stage(config):
    check_inputs(config)
    crnn = CRNN_2Stage(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'], alphabet_size=config['alphabet_size'], rnn_hidden_dim=config["rnn_dimension"],
                       n_rnn=2, leakyRelu=False, recognizer_dropout=config["recognizer_dropout"],
                       online_augmentation=config["online_augmentation"], first_rnn_out_dim=128, rnn_constructor=config["rnn_constructor"])
    return crnn

def create_Nudger(config):
    check_inputs(config)
    crnn = Nudger(rnn_input_dim=config["rnn_input_dimension"], nc=config['num_of_channels'], rnn_hidden_dim=config["rnn_dimension"],
                            rnn_layers=config["nudger_rnn_layers"], leakyRelu=False, rnn_dropout=config["recognizer_dropout"], rnn_constructor=config["rnn_constructor"])
    return crnn

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
        preds_size = Variable(torch.IntTensor([pred_logits.size(0)] * pred_logits.size(1))) # <- what? isn't this square? why are we tiling the size?

        output_batch = pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab
        pred_strs = list(self.decoder.decode_training(output_batch))

        # Get losses
        self.config["logger"].debug("Calculating CTC Loss: {}".format(step))
        loss_recognizer = self.loss_criterion(pred_logits, labels, preds_size, label_lengths)

        # Backprop
        self.config["logger"].debug("Backpropping: {}".format(step))
        self.optimizer.zero_grad()
        loss_recognizer.backward(retain_graph=retain_graph)
        self.optimizer.step()

        loss = torch.mean(loss_recognizer.cpu(), 0, keepdim=False).item()

        # Error Rate
        self.config["stats"]["HWR Training Loss"].accumulate(loss, 1) # Might need to be divided by batch size?
        self.config["logger"].debug("Calculating Error Rate: {}".format(step))
        err, weight = calculate_cer(pred_strs, gt)

        self.config["logger"].debug("Accumulating stats")
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
            self.model.eval()

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
            self.model.eval()

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

class TrainerStrokeRecovery:
    from models import stroke_recovery_loss
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

    def default(self, o):
        return None

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
        line_imgs = item["line_imgs"].to(device)
        label_lengths = item["label_lengths"]
        gt = item["gt_list"]
        suffix = "_train" if train else "_test"

        if train:
            self.model.train()
            self.config.counter.update(epochs=0, instances=line_imgs.shape[0], updates=1)
            #print(self.config.stats[])

        preds = self.eval(line_imgs, self.model)  # This evals and permutes result, Width,Batch,Vocab -> Batch, Width, Vocab

        ## Make predictions relative
        # if self.config.relative_x_pred_abs_eval:
        #     preds = relativefy_batch_torch(preds, reverse=True)  # assume they were in relative positions, convert to absolute

        ## Shorten
        preds = self.truncate(preds, label_lengths) # Convert square torch object to a list, removing predictions related to padding

        loss_tensor, loss = self.loss_criterion.main_loss(preds, gt, label_lengths)

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
    def eval(line_imgs, model):
        """ For offline data, that doesn't have ground truths
        """
        line_imgs = line_imgs.to(device)
        pred_logits = model(line_imgs).cpu()
        return pred_logits.permute(1, 0, 2) # Width,Batch,Vocab -> Batch, Width, Vocab


    def update_stats(self, item, preds, train=True):
        suffix = "_train" if train else "_test"

        ## If not using L1 loss
        if "l1" not in self.loss_criterion.loss_names:
            l1_loss = torch.sum(self.loss_criterion.l1(preds, item["gt_list"], item["label_lengths"]).cpu(), 0, keepdim=False).item()
            self.config.stats["l1"+suffix].accumulate(l1_loss)

        # Don't do the nearest neighbor search by default
        if (self.config.training_nn_loss and train) or (self.config.test_nn_loss and not train) :
            self.config.stats["nn"+suffix].accumulate(self.loss_criterion.calculate_nn_distance(item, preds))