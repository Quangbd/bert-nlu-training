import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from models import JointTinyBert, JointTinyBert2
from optimization import BertAdam
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn

from utils import compute_metrics, get_intent_labels, get_slot_labels


logger = logging.getLogger(__name__)


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        self.teacher_model = JointTinyBert.from_pretrained(args.teacher_model, args, self.intent_label_lst, self.slot_label_lst)
        self.teacher_model.to(self.device)
        if args.stage=='2.1':
            self.student_model = JointTinyBert2.from_pretrained2(args.tinybert,args, self.intent_label_lst, self.slot_label_lst)
        elif args.stage == '2.2':
            self.student_model = JointTinyBert2.from_pretrained(args.student_model,args, self.intent_label_lst, self.slot_label_lst)
        self.student_model.to(self.device)


    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        param_optimizer = list(self.student_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not self.args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)

        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")

        for epoch_ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            self.student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                input_ids, input_mask,segment_ids, intent_label_ids, slot_labels_ids= batch
                if input_ids.size()[0] != self.args.train_batch_size:
                    continue
                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.

                student_intent_logits, student_slot_logits, student_atts, student_reps = self.student_model(input_ids, segment_ids, input_mask,
                                                                                                            is_student=True)
                with torch.no_grad():
                    teacher_intent_logits,teacher_slot_logits, teacher_atts, teacher_reps = self.teacher_model(input_ids, segment_ids, input_mask)

                if not self.args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device),
                                                  teacher_att)

                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                else:
                    cls_intent_loss = soft_cross_entropy(student_intent_logits / self.args.temperature,
                                                         teacher_intent_logits / self.args.temperature)

                    active_loss = input_mask.view(-1) == 1
                    active_student_logits = student_slot_logits.view(-1, self.student_model.num_slot_labels)[active_loss]
                    active_teacher_logits = teacher_slot_logits.view(-1, self.student_model.num_slot_labels)[active_loss]

                    cls_slot_loss = soft_cross_entropy(active_student_logits / self.args.temperature,
                                                       active_teacher_logits / self.args.temperature)

                    loss = cls_intent_loss + cls_slot_loss
                    tr_cls_loss += cls_intent_loss.item() + cls_slot_loss.item()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += intent_label_ids.size(0)
                nb_tr_steps += 1


                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    optimizer.step()
                    #scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % self.args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Batch size = %d", self.args.eval_batch_size)

                    self.student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    result = {}
                    if self.args.pred_distill:
                        result = self.evaluate("dev")
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss

                    result_to_file(result, output_eval_file)

                    if not self.args.pred_distill:
                        save_model = True
                    else:
                        save_model = False

                        if result['slot_f1'] > best_dev_acc:
                            best_dev_acc = result['slot_f1']
                            save_model = True

                    if save_model:
                        logger.info("***** Save model *****")

                        model_to_save = self.student_model.module if hasattr(self.student_model, 'module') else self.student_model

                        # if not args.pred_distill:
                        #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        output_model_file = os.path.join(self.args.output_dir, 'pytorch_model.bin')
                        output_config_file = os.path.join(self.args.output_dir, 'config.json')

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)

                    self.student_model.train()

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.student_model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, intent_label_ids, slot_labels_ids = batch

                intent_logits, slot_logits, _, _ = self.student_model(input_ids, segment_ids, input_mask)
                loss_fct = CrossEntropyLoss()
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                tmp_intent_eval_loss = loss_fct(intent_logits.view(-1, self.student_model.num_intent_labels), intent_label_ids.view(-1))
                if input_mask is not None:
                    active_loss = input_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.student_model.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.student_model.num_slot_labels), slot_labels_ids.view(-1))

                eval_loss += tmp_intent_eval_loss.mean().item()
                eval_loss += slot_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = intent_label_ids.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, intent_label_ids.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = slot_labels_ids.detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, slot_labels_ids.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.student_model = JointTinyBert2.from_pretrained(self.args.output_dir,self.args, self.intent_label_lst, self.slot_label_lst)
            self.student_model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")