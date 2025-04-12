import torch
import torch.optim
import torch.nn as nn
import numpy as np
import glob
import time
import os
import torch.utils.data.distributed
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil

from data.Load_Data import *
from models.latr import LATR
from experiments.gpu_utils import is_main_process
from utils.utils import *
from utils import eval_3D_lane, eval_3D_once, eval_3D_lane_apollo

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .ddp import *
import os
from .gpu_utils import gpu_available
from utils.logger import * 

class Runner:
    def __init__(self, args):
        self.args = args
        self.logger = init_logger(name=args.experiment_name, log_file=os.path.join(self.args.output_dir,args.experiment_name)+f'/{args.experiment_name}.log')
        # Check GPU availability
        if is_main_process():
            if not gpu_available():
                raise Exception("No GPU available")
            if int(os.getenv('WORLD_SIZE', 1)) >= 1:
                self.logger.info("Let's use %s" % os.environ['WORLD_SIZE'] + "GPUs!")
                torch.cuda.empty_cache()
        
        # Get Dataset
        if is_main_process():
            print("Loading Dataset...")
        if args.is_train:
            self.train_dataset, self.train_loader, self.train_sampler = self._get_train_dataset()
            if is_main_process():
                print("Train Dataset Loaded")

        self.valid_dataset, self.valid_loader, self.valid_sampler = self._get_valid_dataset()
        if is_main_process():
            print("Valid Dataset Loaded")

        # TODO: args for evaluator need to be modified
        if 'openlane' in args.dataset_name:
            self.evaluator = eval_3D_lane.LaneEval(args, logger=self.logger)
        elif 'apollo' in args.dataset_name:
            self.evaluator = eval_3D_lane_apollo.LaneEval(args, logger=self.logger)
        elif 'once' in args.dataset_name:
            self.evaluator = eval_3D_once.LaneEval()
        else:
            assert False
        
        # Tensorboard writer
        if not args.no_tb and is_main_process():
            tensorboard_path = os.path.join(self.args.output_dir, args.experiment_name, 'Tensorboard/')
            mkdir_if_missing(tensorboard_path)
            self.writer = SummaryWriter(tensorboard_path)
        
        if is_main_process():
            self.logger.info("Init Done!")
        
        self.is_apollo = False
        if 'apollo' in args.dataset_name:
            self.is_apollo = True

    def train(self):
        args = self.args

        train_loader = self.train_loader
        train_sampler = self.train_sampler

        global lowest_loss, best_f1_epoch, best_val_f1, best_epoch

        model, optimizer, scheduler, best_epoch,\
            lowest_loss, best_f1_epoch, best_val_f1 = self._get_model_ddp()
        self._log_model_info(model)

        def save_cur_ckpt(loss, with_eval=True, eval_stats=None):
            # Save model
            if not with_eval:
                self.save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, False, epoch+1, os.path.join(self.args.output_dir,args.experiment_name))
            else:
                total_score = loss.item()
                if is_main_process():
                    # File to keep latest epoch
                    with open(os.path.join(args.output_dir, 'first_run.txt'), 'w') as f:
                        f.write(str(epoch + 1))
                global best_val_f1, best_f1_epoch, lowest_loss, best_epoch
                to_copy, to_save = False, True # False if args.save_best else True

                if total_score < lowest_loss:
                    best_epoch = epoch + 1
                    lowest_loss = total_score
                if eval_stats[0] > best_val_f1:
                    to_copy = True
                    best_f1_epoch = epoch + 1
                    best_val_f1 = eval_stats[0]
                    to_save = True
                self.log_eval_stats(eval_stats)
                self.logger.info("===> Last best F1 was {:.8f} in epoch {}".format(best_val_f1, best_f1_epoch))
                if not to_save:
                    return
                self.save_checkpoint({
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, to_copy, epoch+1, os.path.join(self.args.output_dir,args.experiment_name))
                
        for epoch in range(self.args.start_epoch, args.nepochs):
            if is_main_process():
                self.logger.info("\n => Start train set for EPOCH {}".format(epoch + 1))
                self.logger.info('lr is set to {}'.format(optimizer.param_groups[0]['lr']))
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # Define container objects to keep track of multiple losses/metrics
            batch_time = AverageMeter()
            data_time = AverageMeter()         # compute FPS
            epoch_time = AverageMeter()
            
            loss = 0

            # Specify operation modules
            model.train()
            # compute timing
            end = time.time()
            epoch_time.start = end
            # Start training loop
            train_pbar = tqdm(total=len(train_loader), ncols=60)
        
            for i, extra_dict in enumerate(train_loader):
                train_pbar.update(1)
                data_time.update(time.time() - end)
                if gpu_available():
                    json_files = extra_dict.pop('idx_json_file')
                    for k,v in extra_dict.items():
                        extra_dict[k] = v.cuda()
                    image = extra_dict['image']
                image = image.contiguous().float()

                optimizer.zero_grad()
                output = model(image=image, extra_dict=extra_dict, is_training=True)

                loss, loss_info = self._log_training_loss(
                    output, epoch, step=i, data_loader=train_loader)
                
                train_pbar.set_postfix(loss=loss.item())
                if is_main_process():
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                
                loss.backward()
                if args.clip_grad_norm != 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                optimizer.step()

                

                # Time trainig iteration
                batch_time.update(time.time() - end)
                end = time.time()

                # Print info
                if (i + 1) % args.print_freq == 0 and is_main_process():
                    self.logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Batch Time / Avg Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss:.8f} {loss_info}'.format(
                            epoch+1, i+1, len(train_loader), 
                            batch_time=batch_time, data_time=data_time,
                            loss=loss.item(), loss_info=loss_info))
            scheduler.step()
            train_pbar.close()

            epoch_time.update(time.time() - epoch_time.start)

            if is_main_process():
                self.logger.info('Epoch time : {:.3f} hours.'.format(epoch_time.val / 60 / 60))

            meet_eval_freq = args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0
            last_ep = (epoch == args.nepochs - 1)

            if meet_eval_freq or last_ep:
                loss_valid_list, eval_stats = self.validate(model)
                if eval_stats[0] >= best_val_f1:
                    self.logger.info(' >>> to save new best model at ep : %s with F1 %s' % ((epoch+1), eval_stats[0]))
                    save_cur_ckpt(loss, with_eval=True, eval_stats=eval_stats)
                elif last_ep:
                    self.logger.info(' >>> to save the last model at ep : %s with F1 %s' % ((epoch+1), eval_stats[0]))
                    save_cur_ckpt(loss, with_eval=True, eval_stats=eval_stats)
                else:
                    self.logger.info(' >>> skip model at ep : %s with lower F1 : %s' % ((epoch+1), eval_stats[0]))
            
                self.log_eval_stats(eval_stats)
            #TODO: fix the eval stats against  torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate more than 1EB memory.
                # save_cur_ckpt(loss, with_eval=False)
            dist.barrier()
            torch.cuda.empty_cache()

        # at the end of training
        if not args.no_tb and is_main_process():
            self.writer.close()


    def _log_training_loss(self, output, epoch, step, data_loader):
        loss = 0.0
        loss_info = ''
        for k, v in output.items():
            if 'loss' in k:
                loss = loss + v
                # print(k, v)
                loss_info = loss_info + '| %s:%.4f ' % (k, v.item() if isinstance(v, torch.Tensor) else v)
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if is_main_process():
                    self.writer.add_scalar(k, v, epoch*len(data_loader) + step)
        return loss, loss_info
    def validate(self, model, **kwargs):
        args = self.args
        loader = self.valid_loader

        pred_lines_sub=[]
        gt_lines_sub=[]

        model.eval()
        
        with torch.no_grad():
            val_pbar = tqdm(total=len(loader), ncols=50)
            for i, extra_dict in enumerate(loader):
                val_pbar.update(1)

                if not args.no_cuda:
                    json_files = extra_dict.pop('idx_json_file')
                    for k,v in extra_dict.items():
                        extra_dict[k] = v.cuda()
                    image = extra_dict['image']
                image = image.contiguous().float()
                output = model(image=image, extra_dict=extra_dict, is_training=False)
                all_line_preds = output["all_line_preds"]
                all_cls_scores = output["all_cls_scores"]

                all_line_preds = all_line_preds[-1]
                all_cls_scores = all_cls_scores[-1]
                num_el = all_cls_scores.shape[0]
                if 'cam_extrinsics' in extra_dict:
                    cam_extrinsics_all = extra_dict['cam_extrinsics']
                    cam_intrinsics_all = extra_dict['cam_intrinsics']
                else:
                    cam_extrinsics_all, cam_intrinsics_all = None, None

                # Print info
                if (i + 1) % args.print_freq == 0 and is_main_process():
                    self.logger.info('Test: [{0}/{1}]'.format(i+1, len(loader)))
                    # Write results
                for j in range(num_el):
                    json_file = json_files[j]
                    if cam_extrinsics_all is not None:
                        extrinsic = cam_extrinsics_all[j].cpu().numpy()
                        intrinsic = cam_intrinsics_all[j].cpu().numpy()

                    with open(json_file, 'r') as file:
                        if 'apollo' in args.dataset_name:
                            json_line = json.loads(file.read())
                            if 'extrinsic' not in json_line:
                                json_line['extrinsic'] = extrinsic
                            if 'intrinsic' not in json_line:
                                json_line['intrinsic'] = intrinsic
                        else:
                            file_lines = [line for line in file]
                            json_line = json.loads(file_lines[0])
                        
                    json_line['json_file'] = json_file
                    if 'once' in args.dataset_name:
                        if 'train' in json_file:
                            img_path = json_file.replace('train', 'data').replace('.json', '.jpg')
                        elif 'val' in json_file:
                            img_path = json_file.replace('val', 'data').replace('.json', '.jpg')
                        elif 'test' in json_file:
                            img_path = json_file.replace('test', 'data').replace('.json', '.jpg')
                        json_line["file_path"] = img_path

                    gt_lines_sub.append(copy.deepcopy(json_line))

                    lane_pred = all_line_preds[j].cpu().numpy()
                    cls_pred = torch.argmax(all_cls_scores[j], dim=-1).cpu().numpy()
                    pos_lanes = lane_pred[cls_pred > 0]

                    if self.args.num_category > 1:
                        scores_pred = torch.softmax(all_cls_scores[j][cls_pred>0], dim=-1).cpu().numpy()
                    else:
                        scores_pred = torch.sigmoid(all_cls_scores[j][cls_pred>0]).cpu().numpy()

                    if pos_lanes.shape[0]:
                        lanelines_pred = []
                        lanelines_prob = []
                        xs = pos_lanes[:, 0:args.num_y_steps]
                        ys = np.tile(args.anchor_y_steps.copy()[None, :], (xs.shape[0], 1))
                        zs = pos_lanes[:, args.num_y_steps:2*args.num_y_steps]
                        vis = pos_lanes[:, 2*args.num_y_steps:]

                        for tmp_idx in range(pos_lanes.shape[0]):
                            cur_vis = vis[tmp_idx] > 0
                            cur_xs = xs[tmp_idx][cur_vis]
                            cur_ys = ys[tmp_idx][cur_vis]
                            cur_zs = zs[tmp_idx][cur_vis]

                            if cur_vis.sum() < 2:
                                continue

                            lanelines_pred.append([])
                            for tmp_inner_idx in range(cur_xs.shape[0]):
                                lanelines_pred[-1].append(
                                    [cur_xs[tmp_inner_idx],
                                     cur_ys[tmp_inner_idx],
                                     cur_zs[tmp_inner_idx]])
                            lanelines_prob.append(scores_pred[tmp_idx].tolist())
                    else:
                        lanelines_pred = []
                        lanelines_prob = []
                    # if is_main_process():
                    #     print("lanelines_pred", lanelines_pred)
                    #     print("lanelines_prob", lanelines_prob)

                    json_line["pred_laneLines"] = lanelines_pred
                    json_line["pred_laneLines_prob"] = lanelines_prob

                    pred_lines_sub.append(copy.deepcopy(json_line))
                    img_path = json_line['file_path']
                    
                    if args.dataset_name == 'once':
                        self.save_eval_result_once(args, img_path, lanelines_pred, lanelines_prob)
            val_pbar.close()

            if 'openlane' in args.dataset_name:
                eval_stats = self.evaluator.bench_one_submit_ddp(
                    pred_lines_sub, gt_lines_sub, args.model_name,
                    args.pos_threshold, vis=False)
            elif 'once' in args.dataset_name:
                eval_stats = self.evaluator.lane_evaluation(
                    args.data_dir + 'val', '%s/once_pred/test' % (args.output_dir),
                    args.eval_config_dir, args)
            elif 'apollo' in args.dataset_name:
                self.logger.info(' >>> eval mAP | [0.05, 0.95]')
                eval_stats = self.evaluator.bench_one_submit_ddp(
                    pred_lines_sub, gt_lines_sub,
                    args.model_name, args.pos_threshold, vis=False)
            else:
                assert False
                
            if any(name in args.dataset_name for name in ['openlane', 'apollo']):
                gather_output = [None for _ in range(args.world_size)]
                # all_gather all eval_stats and calculate mean
                dist.all_gather_object(gather_output, eval_stats)
                dist.barrier()
                eval_stats = self._recal_gpus_val(gather_output, eval_stats)

                loss_list = []
                return loss_list, eval_stats
            elif 'once' in args.dataset_name:
                loss_list = []
                return loss_list, eval_stats

    def _recal_gpus_val(self, gather_output, eval_stats):
        args = self.args

        apollo_metrics = {
            'r_lane': 0, 
            'p_lane': 0, 
            'cnt_gt': 0, 
            'cnt_pred': 0
        }
        openlane_metrics = {
            'r_lane': 0, 
            'p_lane': 0, 
            'c_lane': 0, 
            'cnt_gt': 0, 
            'cnt_pred': 0,
            'match_num': 0
        }

        if 'apollo' in self.args.dataset_name:
            # apollo no category accuracy.
            start_idx = 7
            gather_metrics = apollo_metrics
        else:
            start_idx = 8
            gather_metrics = openlane_metrics
        
        for i, k in enumerate(gather_metrics.keys()):
            gather_metrics[k] = np.sum(
                [eval_stats_sub[start_idx + i] for eval_stats_sub in gather_output])

        if gather_metrics['cnt_gt']!=0 :
            Recall = gather_metrics['r_lane'] / gather_metrics['cnt_gt']
        else:
            Recall = gather_metrics['r_lane'] / (gather_metrics['cnt_gt'] + 1e-6)
        if gather_metrics['cnt_pred'] !=0 :
            Precision = gather_metrics['p_lane'] / gather_metrics['cnt_pred']
        else:
            Precision = gather_metrics['p_lane'] / (gather_metrics['cnt_pred'] + 1e-6)
        if (Recall + Precision)!=0:
            f1_score = 2 * Recall * Precision / (Recall + Precision)
        else:
            f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
        
        if 'apollo' not in self.args.dataset_name:
            if gather_metrics['match_num']!=0:
                category_accuracy = gather_metrics['c_lane'] / gather_metrics['match_num']
            else:
                category_accuracy = gather_metrics['c_lane'] / (gather_metrics['match_num'] + 1e-6)
        
        eval_stats[0] = f1_score
        eval_stats[1] = Recall
        eval_stats[2] = Precision
        if self.is_apollo:
            err_start_idx = 3
        else:
            eval_stats[3] = category_accuracy
            err_start_idx = 4
        for i in range(4):
            err_idx = err_start_idx + i
            eval_stats[err_idx] = np.sum([eval_stats_sub[err_idx] for eval_stats_sub in gather_output]) / args.world_size
        return eval_stats



    def log_eval_stats(self, eval_stats):
        if self.is_apollo:
            return self._log_genlane_eval_info(eval_stats)

        if is_main_process():
            self.logger.info("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
            self.logger.info("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
            self.logger.info("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
            self.logger.info("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
            self.logger.info("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
            self.logger.info("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
            self.logger.info("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
            self.logger.info("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))

    def _log_model_info(self, model, **kwargs):
        args = self.args
        if not is_main_process():
            return
        self.logger.info(40*"="+"\nArgs:{}\n".format(args)+40*"=")
        self.logger.info("Init model: '{}'".format(args.mod))
        self.logger.info("Number of parameters in model {} is {:.3f}M".format(args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))
    
        
    def save_checkpoint(self, state, to_copy, epoch, save_path):
        if is_main_process():
            self.logger.info('Saving checkpoint to {}'.format(save_path))

            if to_copy:
                file_pre = f'model_best_epoch_{epoch}.pth.tar'
                self.logger.info('save the best model : %s' % epoch)
            else:
                file_pre = f'checkpoint_model_epoch_{epoch}.path.tar'

            filepath = os.path.join(save_path, file_pre)
            torch.save(state, filepath)





    def _get_model_ddp(self):
        args = self.args
        model = LATR(args)

        if is_main_process():
            self.logger.info("Convert model with Sync BatchNorm")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if gpu_available():
            device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
            model = model.to(device)
            print(f"model to {device}")

        model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, \
            optim_saved_state, schedule_saved_state = self.resume_model(model)
        dist.barrier()
        if args.distributed:
            self.logger.info("===> DDP init")
            model = DDP(
                model, device_ids=[int(os.environ['LOCAL_RANK'])],
                output_device=int(os.environ['LOCAL_RANK']),
                find_unused_parameters=True
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.start_epoch+args.nepochs)
        if optim_saved_state is not None:
            if is_main_process():
                self.logger.info("Loading optimizer state from checkpoint")
            optimizer.load_state_dict(optim_saved_state)
        if schedule_saved_state is not None:
            if is_main_process():
                self.logger.info("Loading scheduler state from checkpoint")
            scheduler.load_state_dict(schedule_saved_state)
        self.logger.info("==> DDP init finished")
        return model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1

    def resume_model(self, model, path=''):
        args = self.args

        best_epoch = 0
        lowest_loss = np.inf
        best_f1_epoch = 0
        best_val_f1 = -1e-5
        optim_saved_state = None
        schedule_saved_state = None

        if len(path) == 0 and args.resume:
            path = os.path.join(os.path.join(self.args.output_dir,args.experiment_name), 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
            if not os.path.isfile(path):
                print('No checkpoint found at {}'.format(path))
                path = os.path.join(os.path.join(self.args.output_dir,args.experiment_name), f'model_best_epoch_{args.resume}.pth.tar')
            
        if os.path.isfile(path):
            self.logger.info("=> loading checkpoint from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            if is_main_process():
                model.load_state_dict(checkpoint['state_dict'])
                self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))

            optim_saved_state = checkpoint['optimizer']
            schedule_saved_state = checkpoint['schedule']

            args.start_epoch = int(args.resume)
        else:
            if is_main_process():
                self.logger.info("=> Warning: no checkpoint found at '{}'".format(path))
        
        return model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, optim_saved_state, schedule_saved_state

    def _get_model_from_cfg(self):
        args = self.args
        model = LATR(args)
        
        if args.sync_bn:
            if is_main_process():
                self.logger.info("Convert model with Sync BatchNorm")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
        if gpu_available():
            device = torch.device("cuda", int(os.environ['LOCAL_RANK']))
            model = model.to(device)

        return model

    def _load_ckpt_from_workdir(self, model):
        args = self.args
        if args.eval_ckpt:
            best_file_name = args.eval_ckpt
        else:
            best_file_name = glob.glob(os.path.join(os.path.join(self.args.output_dir,args.experiment_name)+"/models", 'model_best*'))
            if len(best_file_name) > 0:
                best_file_name = best_file_name[0]
            else:
                best_file_name = ''
        if os.path.isfile(best_file_name):
            checkpoint = torch.load(best_file_name)
            if is_main_process():
                self.logger.info("=> loading checkpoint '{}'".format(best_file_name))
                model.load_state_dict(checkpoint['state_dict'])
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(best_file_name))
    def eval(self):
        self.logger.info('>>>>>  start eval <<<<< \n')
        args = self.args

        model = self._get_model_from_cfg()
        self._load_ckpt_from_workdir(model)
        dist.barrier()
        # DDP setting
        if args.distributed:
            model = DDP(
                model, device_ids=[int(os.environ['LOCAL_RANK'])],
                output_device=int(os.environ['LOCAL_RANK']),
                find_unused_parameters=True
            )
        _, eval_stats = self.validate(model)

        if is_main_process() and (eval_stats is not None):
            self.log_eval_stats(eval_stats)

    def _get_train_dataset(self):
        args = self.args
        if 'openlane' in args.dataset_name:
            train_dataset = LaneDataset(args.dataset_base_dir, args.json_file_path + 'training/', args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
        return train_dataset, train_loader, train_sampler


    def _get_valid_dataset(self):
        args = self.args
        if 'openlane' in args.dataset_name:
            if not args.evaluate_case:
                valid_dataset = LaneDataset(args.dataset_base_dir, args.json_file_path + 'validation/', args)
            else:
                # note: for case eval, change the 'up_down_case' to one of case names.['up_down_case','curve_case','extreme_weather_case','intersection_case','merge_split_case','night_case']  
                valid_dataset = LaneDataset(args.dataset_base_dir, args.json_file_path + 'test/up_down_case/', args)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers)
        return valid_dataset, valid_loader, valid_sampler
    def log_eval_stats(self, eval_stats):
        if self.is_apollo:
            return self._log_genlane_eval_info(eval_stats)

        if is_main_process():
            self.logger.info("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
            self.logger.info("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
            self.logger.info("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
            self.logger.info("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
            self.logger.info("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
            self.logger.info("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
            self.logger.info("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
            self.logger.info("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))


    def _log_genlane_eval_info(self, eval_stats):
        if is_main_process():
            self.logger.info("===> Evaluation on validation set: \n"
                "laneline F-measure {:.8} \n"
                "laneline Recall  {:.8} \n"
                "laneline Precision  {:.8} \n"
                "laneline x error (close)  {:.8} m\n"
                "laneline x error (far)  {:.8} m\n"
                "laneline z error (close)  {:.8} m\n"
                "laneline z error (far)  {:.8} m\n".format(eval_stats[0], eval_stats[1],
                                                            eval_stats[2], eval_stats[3],
                                                            eval_stats[4], eval_stats[5],
                                                            eval_stats[6])) 