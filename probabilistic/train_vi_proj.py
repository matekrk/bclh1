#!/usr/bin/env python3
# Copyright 2020 Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :probabilistic/train_vi.py
# @author         :mc, ch
# @contact        :mariacer@ethz.ch
# @created        :12/15/2020
# @version        :1.0
# @python_version :3.6.9
"""
Using Variational Inference for Prior-Focused and Posterior-Replay CL
---------------------------------------------------------------------

The module :mod:`probabilistic.train_vi` contains training, testing and
evaluation functions for a probabilistic classifier in a continual learning
setting. I.e., given a sequence of tasks, the goal is to obtain a Bayesian
Neural Network that performs well and provides meaningful predictive
uncertainties on all tasks (see also module
:mod:`probabilistic.prob_mnist.train_bbb`).

Specifically, this module uses one of two VI algorithms to learn an implicit
weight posterior, that is realized through a hypernetwork. When using
prior-focused CL, the implicit distribution (hypernetwork) from the previous
task is used as prior.

When learning a posterior per task, the implicit distribution in the
hypernetwork is protected via a hyper-hypernetwork.

Two VI training algorithms that deal with implicit distributions. One option is
the algorithm AVB proposed in

    Mescheder et al., "Adversarial Variational Bayes: Unifying Variational
    Autoencoders and Generative Adversarial Networks", 2018.
    https://arxiv.org/abs/1701.04722

and another one is SSGE proposed in

    Shi, Jiaxin, Shengyang Sun, and Jun Zhu. "A spectral approach to gradient 
    estimation for implicit distributions." ICML, 2018.
    https://arxiv.org/abs/1806.02925

which directly estimates the gradients of the log-density without having to
estimate the density itself.
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from warnings import warn

from mnets.classifier_interface import Classifier
from hnets.hnet_helpers import get_conditional_parameters
from probabilistic import ewc_utils as ewcutil
from probabilistic.prob_cifar import train_utils as pcutils
from probabilistic.prob_gmm import train_utils as pgutils
from probabilistic.prob_mnist import train_utils as pmutils
from probabilistic.regression import train_avb as regavb
from probabilistic import ssge_utils as ssgeu
from utils import hnet_regularizer as hreg
from utils import gan_helpers as gan
from utils import torch_utils as tutils
from utils import sim_utils as sutils

def test(data_handlers, mnet, hnet, hhnet, device, config, shared, logger,
         writer, test_ids=None, method='avb', verbose=True, 
         fixed_embedding=-1, optimal_embedding=-1, projection=True):
    """Test the performance on all tasks.

    Additionally when we fix embedding to be the first one.

    Note:
        To infer task identity, each tested task (i.e., dataset) is processed
        by all task-specific models.

    Note:
        As long as one hasn't trained on all tasks, the accuracies based on
        task inference are hard to interpret, since task inference becomes
        harder the more tasks are involved (e.g., if there is only one task
        then task inference always picks the correct task ID).

    Args:
        (....): See docstring of function :func:`train`.
        data_handlers: A list of data handlers, each representing a task.

            Note:
                The indices ``list(range(len(data_handlers)))`` will determine
                the task-conditioned models considered for task inference
                (i.e., the tested task will be fed through all these models to
                determine the per-task uncertainty).

            Note:
                Should only contain tasks that have been trained on. It doesn't
                make sense to feed an input through an untrained model and it
                also doesn't make sense to determine the uncertainty from an
                untrained dataset, since we can't compare it to its
                in-distribution uncertainty.
        test_ids (list, optional): List of task IDs, that should be tested on.
            If not provided, all tasks contained in ``data_handlers`` are
            tested. Otherwise, only the indices in this option are considered.

            Note:
                If provided, it has to contain ``len(data_handlers) - 1``, as we
                assume this is the last task were training was performed on
                (only matter for updating the performance summary).
        method (str): Either ``'avb'``, ``'bbb'``, ``'ewc'`` or ``'ssge'``.
            Whether networks trained via :mod:`probabilistic.train_vi` using
            SSGE or AVB are tested or networks trained via
            :mod:`probabilistic.prob_mnist.train_bbb` or networks trained via
            :mod:`probabilistic.ewc_utils.train`.
        verbose (bool, optional): If ``True``, task inference will be reported
            task by task.
        fixed_embedding (bool, optional): If not ``-1``, task inference will be
            reported also with the fixed embedding, which has to be provided.
        optimal_embedding (int, optional): If ``-1``, the default correspondance
            task_id <-> embedding_id will be used.
    """
    # FIXME This function became quite long, maybe one should put some effort
    # into disentangling it.
    logger.info('### Testing all trained tasks ... ###')

    assert method in ['avb', 'bbb', 'ewc', 'ssge', 'mt']

    if test_ids is not None:
        assert np.all(np.array(test_ids) >= 0) and \
               np.all(np.array(test_ids) < len(data_handlers))
        test_ids = np.sort(np.unique(test_ids))
        logger.debug('Testing only on tasks: %s.' % (test_ids+1))
        test_ids = test_ids.tolist()
    else:
        test_ids = list(range(len(data_handlers)))

    if hnet is None and config.cl_scenario == 2 and \
            not config.train_from_scratch:
        # No task-conditioned weights and single output head.
        warn('Task inference calculated in test method doesn\'t make any ' +
             'sense, since the main network cannot be conditioned on task ' +
             'identity.')
    if method == 'avb' and shared.prior_focused and \
            config.cl_scenario == 2 and not config.train_from_scratch:
        warn('Uncertainty-based task-inference doesn\'t make sense for ' +
             'prior-focused methods with only a single head.')

    pcutils.set_train_mode(False, mnet, hnet, hhnet, None)

    n = len(data_handlers)
    n_testids = len(test_ids)

    # Whether we tested on all tasks trained so far.
    tested_all = True
    if len(test_ids) < n:
        tested_all = False
        # Otherwise we won't be able to set the during accuracies in the
        # performance summary correctly.
        assert n-1 in test_ids

    # Current accuracy per task (assuming correct task embedding being
    # provided).
    acc_task_given = np.ones(n) * -1.
    # Task accuracy if inferred embeddings are used to process samples
    # (using inference based on entropy / confidence / model agreement).
    acc_task_inferred_ent = np.ones(n) * -1.
    acc_task_inferred_conf = np.ones(n) * -1.
    acc_task_inferred_agree = np.ones(n) * -1.
    # Task inference accuracy per task, i.e., how often was the correct task
    # embedding chosen (based on entropy / confidence / model agreement).
    task_inference_acc_ent = np.ones(n) * -1.
    task_inference_acc_conf = np.ones(n) * -1.
    task_inference_acc_agree = np.ones(n) * -1.


    # Task accuracy for fixed embedding is always used to process samples
    # (additionally keep track of entropy / confidence / model agreement).
    # (additionally calculate in-distri and OOD entropies)
    if fixed_embedding != -1:
        gen_acc_task_fixemb = np.ones(n) * -1.
        gen_ent_task_fixemb = np.ones(n) * -1.
        gen_conf_task_fixemb = np.ones(n) * -1
        gen_agree_task_fixemb = np.ones(n) * -1
        acc_task_fixemb = np.ones(n) * -1.
        ent_task_fixemb = np.ones(n) * -1.
        conf_task_fixemb = np.ones(n) * -1
        agree_task_fixemb = np.ones(n) * -1

    # Average in-distribution entropies per task.
    in_ents = np.ones(n) * -1.
    # Average out-of-distribution entropies per task.
    out_ents = np.ones(n) * -1.

    with torch.no_grad():
        normal_post = None
        if method == 'ewc':
            assert hnet is None
            normal_post = ewcutil.build_ewc_posterior(data_handlers, mnet,
                device, config, shared, logger, writer, n, task_id=n-1)

        if config.train_from_scratch and n > 1:
            # We need to iterate over different networks when we want to
            # measure the uncertainty of dataset i on task j.
            # Note, we will always load the corresponding checkpoint of task j
            # before using these networks.
            assert not method == 'mt'
            if method == 'avb' or method == 'ssge' or method == 'ewc':
                mnet_other, hnet_other, hhnet_other, _ = \
                    pcutils.generate_networks(config, shared, logger,
                        shared.all_dhandlers, device, create_dis=False)
            else:
                hhnet_other = None
                mnet_other, hnet_other = pmutils.generate_gauss_networks(config,
                    logger, shared.all_dhandlers, device,
                    shared.experiment_type, no_mnet_weights=None,
                    create_hnet=hnet is not None)

            pcutils.set_train_mode(False, mnet_other, hnet_other, hhnet_other,
                                   None)

        task_n_mnet = mnet
        task_n_hnet = hnet
        task_n_hhnet = hhnet
        task_n_normal_post = normal_post

        # This renaming is just a protection against myself, that I don't use
        # any of those networks (`mnet`, `hnet`, `hhnet`) in the future
        # inside the loop when training from scratch.
        # Note, I reset those values at the end of the loop.
        if config.train_from_scratch:
            mnet = None
            hnet = None
            hhnet = None
            normal_post = None

        ### For each data set (i.e., for each task).
        for i in test_ids:
            data = data_handlers[i]

            ### Assuming hhnet, which embedding is the one that was trained on i data
            # Ground truth, only to compute "given" in 3rd scenario
            optimal_embedding_task = i if optimal_embedding == -1 else optimal_embedding

            j_range = range(n) # if config.fixed_embedding == -1 else [config.fixed_embedding] #TODO

            ### Task inference.
            # We need to iterate over each task embedding and measure the
            # predictive uncertainty (entropy) in order to decide which
            # embedding to use.
            # Entropy of predictive distribution of all models on current data.
            entropies = np.empty((data.num_test_samples, len(j_range)))
            # Confidence refers to highest prob. in predictive distribution.
            confidence = np.empty((data.num_test_samples, len(j_range)))
            # Agreement measures the deviation in individual model predictions
            # (i.e., deviation in predictions for different weight samples).
            agreement = np.empty((data.num_test_samples, len(j_range)))
            labels = np.empty((data.num_test_samples))
            pred_labels = np.empty((data.num_test_samples, len(j_range)))

            ## Placeholders for fixed embedding
            if fixed_embedding != -1:
                gen_samples_ent_task_fixemb = np.empty((data.num_test_samples, len(j_range)))
                gen_samples_conf_task_fixemb = np.empty((data.num_test_samples, len(j_range)))
                gen_samples_agree_task_fixemb = np.empty((data.num_test_samples, len(j_range)))
                gen_samples_pred_labels_fixemb = np.empty((data.num_test_samples, len(j_range)))
                samples_ent_task_fixemb = np.empty((data.num_test_samples, len(j_range)))
                samples_conf_task_fixemb = np.empty((data.num_test_samples, len(j_range)))
                samples_agree_task_fixemb = np.empty((data.num_test_samples, len(j_range)))
                samples_pred_labels_fixemb = np.empty((data.num_test_samples, len(j_range)))

            for j in j_range:
                ckpt_score_j = None
                if config.train_from_scratch and j == (n-1):
                    # Note, the networks trained on dataset (n-1) haven't been
                    # checkpointed yet.
                    mnet_j = task_n_mnet
                    hnet_j = task_n_hnet
                    hhnet_j = task_n_hhnet
                    normal_post_j = task_n_normal_post
                elif config.train_from_scratch:
                    ckpt_score_j = pmutils.load_networks(shared, j, device,
                        logger, mnet_other, hnet_other, hhnet=hhnet_other,
                        dis=None)
                    mnet_j = mnet_other
                    hnet_j = hnet_other
                    hhnet_j = hhnet_other
                    normal_post_j = None
                    if method == 'ewc':
                        normal_post_j = ewcutil.build_ewc_posterior( \
                            data_handlers, mnet_j, device, config, shared,
                            logger, writer, n, task_id=j)
                else:
                    mnet_j = mnet
                    hnet_j = hnet
                    hhnet_j = hhnet
                    normal_post_j = normal_post

                inspect_first = 10

                # Compute accuracy of net from task j on data from task i.
                if method == 'avb' or method == 'ssge':
                    gen_acc, gen_ret_vals = pcutils.compute_acc(i, data, mnet_j, hnet_j,
                        hhnet_j, device, config, shared, split_type='test',
                        return_dataset=False, return_entropies=True,
                        return_confidence=True, return_agreement=True,
                        return_pred_labels=True, return_labels=True,
                        deterministic_sampling=True, which_cond_id=j,
                        inspect_first=inspect_first, return_wnorms=True, 
                        return_sim_layers=True, logger=logger, projection=projection, projection_aware=-1)
                    
                    acc, ret_vals = pcutils.compute_acc(i, data, mnet_j, hnet_j,
                        hhnet_j, device, config, shared, split_type='test',
                        return_dataset=False, return_entropies=True,
                        return_confidence=True, return_agreement=True,
                        return_pred_labels=True, return_labels=True,
                        deterministic_sampling=True, which_cond_id=j,
                        inspect_first=inspect_first, return_wnorms=True, 
                        return_sim_layers=True, logger=logger, projection=projection, projection_aware=j)
                else:
                    disable_lrt = config.disable_lrt_test if \
                        hasattr(config, 'disable_lrt_test') else False
                    acc, ret_vals = pmutils.compute_acc(j, data, mnet_j,
                    hnet_j, device, config, shared, split_type='test',
                    return_dataset=False, return_entropies=True,
                    return_confidence=True, return_agreement=True,
                    return_pred_labels=True, return_labels=True,
                    deterministic_sampling=True, disable_lrt=disable_lrt, 
                    normal_post=normal_post_j, which_cond_id=j,
                    inspect_first=inspect_first, return_wnorms=True,
                    return_sim_layers=True)

                """
                # Plot current task uncertainty for toy example.
                if shared.experiment_type.startswith('gmm'):
                    pgutils.plot_gmm_preds(i, data, mnet_j, hnet_j, hhnet_j,
                        device, config, shared, logger, writer, n,
                        draw_samples=n-1==i, normal_post=normal_post_j, which_cond_id=j)
                """

                if j == fixed_embedding:

                    logger.debug(f'Test: Task {i+1} - (fixed) Embedding no projection {j+1}')
                    gen_acc_task_fixemb[i] = gen_acc
                    if verbose:
                        logger.debug('Test: Task %d - Accuracy ' % \
                                    (i+1) + 'using fixed (%d) embedding no projection %.2f%% .' \
                                        % (fixed_embedding, gen_acc_task_fixemb[i]))
                    writer.add_scalar('test/task_%d/gen_acc_fixemb' % i,
                                    gen_acc_task_fixemb[i], n)
                    if verbose:
                        logger.debug('Test: After Task %d - (so far)Mean accuracy using '%\
                                     (i+1) + 'fixed embeddings no projection: %.2f%%.' % acc)
                    writer.add_scalar('test/task_%d/gen_mean_acc_task_fixed' % i,
                                      np.mean(gen_acc_task_fixemb[:i]), n)
                    gen_samples_ent_task_fixemb[:, i] = gen_ret_vals.entropies
                    gen_samples_conf_task_fixemb[:, i] = gen_ret_vals.confidence
                    gen_samples_agree_task_fixemb[:, i] = gen_ret_vals.agreement
                    gen_ent_task_fixemb[i] = samples_ent_task_fixemb[:, i].mean()
                    gen_conf_task_fixemb[i] = samples_conf_task_fixemb[:, i].mean()
                    gen_agree_task_fixemb[i] = samples_agree_task_fixemb[:, i].mean()
                    gen_prediction_confusion_fixemb = gen_ret_vals.prediction_confusion
                    gen_entropies_confusion_fixemb = gen_ret_vals.entropies_confusion
                    if verbose:
                        logger.debug('Confusion predictions: Task %d - confusion matrix of entropies for fixed embedding no projection \n' % \
                                    (i+1) + np.array2string(gen_prediction_confusion_fixemb, precision=2, separator=','))
                    sns.heatmap(gen_prediction_confusion_fixemb, annot=True, cmap='Purples', cbar=False)
                    writer.add_figure('test/task_%d/gen_confusion_predictions_fixemb' % i, plt.gcf(), n)
                    if verbose:
                        logger.debug('Confusion entropies: Task %d - confusion matrix of entropies for fixed embedding no projection \n' % \
                                    (i+1) + np.array2string(gen_entropies_confusion_fixemb, precision=2, separator=','))
                    sns.heatmap(gen_entropies_confusion_fixemb, annot=True, cmap='Purples', cbar=False)
                    writer.add_figure('test/task_%d/gen_confusion_entropies_fixemb' % i, plt.gcf(), n)
                    if verbose:
                        logger.debug('Test: Task %d Fixed embedding no projection - Mean entropy / confidence / ' % \
                                    (i+1) + 'agreement on [potentially] out-dist. samples: ' +
                                    '%.2f / %.2f / %.2f.' \
                                    % (gen_ent_task_fixemb[i], gen_conf_task_fixemb[i], gen_agree_task_fixemb[i]))
                    writer.add_scalar('test/task_%d/gen_ent_fixemb' % i, gen_ent_task_fixemb[i], n)
                    writer.add_histogram('test/task_%d/gen_out_ents_fixemb_hist' % i,
                                        gen_samples_ent_task_fixemb[:, i], n)
                    writer.add_scalar('test/task_%d/gen_conf_fixemb' % i, gen_conf_task_fixemb[i], n)
                    writer.add_histogram('test/task_%d/gen_out_conf_fixemb_hist' % i,
                                        gen_samples_conf_task_fixemb[:, i], n)
                    writer.add_scalar('test/task_%d/gen_agree_fixemb' % i, gen_agree_task_fixemb[i], n)
                    writer.add_histogram('test/task_%d/gen_agree_fixemb_hist' % i,
                                        gen_samples_agree_task_fixemb[:, i], n)
                    gen_labels_fixemb = gen_ret_vals.labels
                    gen_samples_pred_labels_fixemb[:, i] = gen_ret_vals.pred_labels
                    writer.add_histogram('test/task_%d/wnorms_fixemb' % i,
                                        gen_ret_vals.return_wnorms, n)
                    if verbose:
                        logger.debug('Sim layer: Task %d - matrix looks like that \n' % \
                                    (i+1) + np.array2string(gen_ret_vals.sim_layers, precision=2, separator=','))
                    sns.heatmap(gen_ret_vals.sim_layers, annot=True, cmap='Purples', cbar=False)
                    writer.add_figure('test/task_%d/sim_matrix_fixemb' % i, plt.gcf(), n)
                    gen_sum_sim_layers_offdiag = np.sum(gen_ret_vals.sim_layers) - np.sum(np.diagonal(gen_ret_vals.sim_layers))
                    if verbose:
                        logger.debug('Test: Task %d - Sum sim layers off diagonal ' % \
                                    (i+1) + '%.2f .' % (gen_sum_sim_layers_offdiag))
                    writer.add_scalar('test/task_%d/gen_sim_sum_offdiag_fixemb' % i,
                                    gen_sum_sim_layers_offdiag, n)
                    
                    logger.debug(f'Test: Task {i+1} - (fixed) Embedding {j+1}')

                    acc_task_fixemb[i] = acc

                    if verbose:
                        logger.debug('Test: Task %d - Accuracy ' % \
                                    (i+1) + 'using fixed (%d) embedding %.2f%% .' \
                                        % (fixed_embedding, acc_task_fixemb[i]))
                    writer.add_scalar('test/task_%d/acc_fixemb' % i,
                                    acc_task_fixemb[i], n)

                    if verbose:
                        logger.debug('Test: After Task %d - (so far)Mean accuracy using '%\
                                     (i+1) + 'fixed embeddings: %.2f%%.' % acc)
                    writer.add_scalar('test/task_%d/mean_acc_task_fixed' % i,
                                      np.mean(acc_task_fixemb[:i]), n)
                    
                    samples_ent_task_fixemb[:, i] = ret_vals.entropies
                    samples_conf_task_fixemb[:, i] = ret_vals.confidence
                    samples_agree_task_fixemb[:, i] = ret_vals.agreement

                    ent_task_fixemb[i] = samples_ent_task_fixemb[:, i].mean()
                    conf_task_fixemb[i] = samples_conf_task_fixemb[:, i].mean()
                    agree_task_fixemb[i] = samples_agree_task_fixemb[:, i].mean()
                    
                    
                    prediction_confusion_fixemb = ret_vals.prediction_confusion
                    entropies_confusion_fixemb = ret_vals.entropies_confusion

                    if verbose:
                        logger.debug('Confusion predictions: Task %d - confusion matrix of entropies for fixed embedding \n' % \
                                    (i+1) + np.array2string(prediction_confusion_fixemb, precision=2, separator=','))
                    sns.heatmap(prediction_confusion_fixemb, annot=True, cmap='Purples', cbar=False)
                    writer.add_figure('test/task_%d/confusion_predictions_fixemb' % i, plt.gcf(), n)

                    if verbose:
                        logger.debug('Confusion entropies: Task %d - confusion matrix of entropies for fixed embedding \n' % \
                                    (i+1) + np.array2string(entropies_confusion_fixemb, precision=2, separator=','))
                    sns.heatmap(entropies_confusion_fixemb, annot=True, cmap='Purples', cbar=False)
                    writer.add_figure('test/task_%d/confusion_entropies_fixemb' % i, plt.gcf(), n)

                
                    if verbose:
                        logger.debug('Test: Task %d Fixed embedding - Mean entropy / confidence / ' % \
                                    (i+1) + 'agreement on [potentially] out-dist. samples: ' +
                                    '%.2f / %.2f / %.2f.' \
                                    % (ent_task_fixemb[i], conf_task_fixemb[i], agree_task_fixemb[i]))

                    writer.add_scalar('test/task_%d/ent_fixemb' % i, ent_task_fixemb[i], n)
                    writer.add_histogram('test/task_%d/out_ents_fixemb_hist' % i,
                                        samples_ent_task_fixemb[:, i], n)

                    writer.add_scalar('test/task_%d/conf_fixemb' % i, conf_task_fixemb[i], n)
                    writer.add_histogram('test/task_%d/out_conf_fixemb_hist' % i,
                                        samples_conf_task_fixemb[:, i], n)

                    writer.add_scalar('test/task_%d/agree_fixemb' % i, agree_task_fixemb[i], n)
                    writer.add_histogram('test/task_%d/agree_fixemb_hist' % i,
                                        samples_agree_task_fixemb[:, i], n)

                    labels_fixemb = ret_vals.labels
                    samples_pred_labels_fixemb[:, i] = ret_vals.pred_labels

                    # FIXME
                    if config.cl_scenario == 1 or \
                        (config.cl_scenario == 3 and config.split_head_cl3 and config.fixed_embedding == -1):
                        # Note, assuming all tasks have the same output head size.
                        labels_fixemb += i * data.num_classes
                        # realigning
                        labels_fixemb = labels_fixemb % data.num_classes
                    if config.cl_scenario == 1 or \
                            (config.cl_scenario == 3 and config.split_head_cl3 and config.fixed_embedding == -1):
                        # Note, assuming all tasks have the same output head size.
                        samples_pred_labels_fixemb[:, i] += j * \
                            data.num_classes
                    
                    # Norms of sampled networks
                    writer.add_histogram('test/task_%d/wnorms_fixemb' % i,
                                        ret_vals.return_wnorms, n)
                    
                    # Last layer analysis
                    if verbose:
                        logger.debug('Sim layer: Task %d - matrix looks like that \n' % \
                                    (i+1) + np.array2string(ret_vals.sim_layers, precision=2, separator=','))
                    
                    sns.heatmap(ret_vals.sim_layers, annot=True, cmap='Purples', cbar=False)
                    writer.add_figure('test/task_%d/sim_matrix_fixemb' % i, plt.gcf(), n)

                    sum_sim_layers_offdiag = np.sum(ret_vals.sim_layers) - np.sum(np.diagonal(ret_vals.sim_layers))
                    if verbose:
                        logger.debug('Test: Task %d - Sum sim layers off diagonal ' % \
                                    (i+1) + '%.2f .' % (sum_sim_layers_offdiag))
                    writer.add_scalar('test/task_%d/sim_sum_offdiag_fixemb' % i,
                                    sum_sim_layers_offdiag, n)

                    # This is only for fixemb for now
                    if inspect_first:

                        # Data
                        for k in range(ret_vals.inspect_first_pred_j.shape[1]):
                            if verbose:
                               logger.debug(f'Test: Task %d - Mean prediction on first {k}^th image' % \
                                   (i+1) + ' using fixed embedding is %d .' \
                                       % (ret_vals.inspect_first_pred_all[k]))
                            #FIXME: Wandb issue with single histogram, but works with tensorboard
                            writer.add_histogram(f'test/task_%d/first_pred_im-{k}_fixemb' % i,
                                    ret_vals.inspect_first_pred_j[:, k], n, config.num_classes_per_task)

                        # Networks
                        writer.add_histogram(f'test/task_%d/first_acc_nets_fixemb' % i,
                                                ret_vals.inspect_first_acc_j, n)
                        for jj in range(len(ret_vals.inspect_first_acc_j)):
                            # Accuracies
                            if verbose:
                                logger.debug(f'Test: Task %d - Mean accuracy by {jj}-net' % \
                                    (i+1) + ' using fixed embeddings %.2f%% .' \
                                        % (ret_vals.inspect_first_acc_j[jj]))
                            
                            # Predictions
                            writer.add_histogram(f'test/task_%d/first_pred_net-{jj}_fixemb' % i,
                                                ret_vals.inspect_first_pred_j[jj], n, config.num_classes_per_task)

                        # Data
                        if verbose:
                            logger.debug('Test: Task %d - Accuracy on first images' % \
                                        (i+1) + 'using fixed embedding %.2f%% .' \
                                            % (ret_vals.inspect_first_acc_all))
                        writer.add_scalar('test/task_%d/first_acc_fixemb' % i,
                                        ret_vals.inspect_first_acc_all, n)

                        # Standard deviations of nets on the first data
                        for k in range(ret_vals.inspect_first_std.shape[0]):
                            writer.add_histogram(f'test/task_%d/first_im-{k}_std_fixemb' % i,
                                            ret_vals.inspect_first_std[k], n)

                if j == optimal_embedding_task: # I.e., we used the correct embedding/network.
                    
                    logger.debug(f'Test: Task {i+1} - (correct) Embedding {j+1}')
                    # confusion matrices - TAW
                    shared.summary['confusion_matrix_acc_taw'][n_testids-1, i] = acc
                    if i < n_testids-1:
                        shared.summary['confusion_matrix_forg_taw'][n_testids-1, i] = shared.summary['confusion_matrix_acc_taw'][:n_testids-1, i].max(0) - shared.summary['confusion_matrix_acc_taw'][n_testids-1, i]
                    
                    # Sanity check.
                    if ckpt_score_j is not None:
                        assert np.allclose(acc, ckpt_score_j)

                    # Plot all so far
                    if shared.experiment_type.startswith('gmm'):
                        pgutils.plot_gmm_preds_sofa(i, data, mnet_j, hnet_j, hhnet_j,
                            device, config, shared, logger, writer, n,
                            draw_samples=n-1==i, normal_post=normal_post_j,
                            which_cond_id=j)
                        
                    # Plot with projections
                    if shared.experiment_type.startswith('gmm'):
                        pgutils.plot_gmm_preds_proj(i, data, mnet_j, hnet_j, hhnet_j,
                            device, config, shared, logger, writer, n,
                            draw_samples=n-1==i, normal_post=normal_post_j,
                            which_cond_id=j, projection=True)

                    # Plot current task uncertainty for toy example.
                    if shared.experiment_type.startswith('gmm'):
                        pgutils.plot_gmm_preds(i, data, mnet_j, hnet_j, hhnet_j,
                            device, config, shared, logger, writer, n,
                            draw_samples=n-1==i, normal_post=normal_post_j,
                            which_cond_id=j)

                    labels = ret_vals.labels # Labels are the same for every j!

                    if method == 'avb' or method == 'ssge':
                        current_weights = ret_vals.theta
                    elif method == 'ewc' and normal_post is None \
                            or method == 'mt':
                        # True for all EWC runs, even if normal_post is given!
                        current_weights = mnet_j.internal_params
                    else:
                        current_weights = ret_vals.w_mean
                        if ret_vals.w_std is not None and method != 'ewc':
                            current_weights = list(current_weights) + \
                                list(ret_vals.w_std)

                    acc_task_given[i] = acc

                    if verbose:
                        logger.debug('Test: Task %d - Accuracy using correct '%\
                                     (i+1) + 'embeddings: %.2f%%.' % acc)
                    writer.add_scalar('test/task_%d/acc_task_given' % i,
                                      acc, n)

                entropies[:, j] = ret_vals.entropies
                # Whiten entropies.
                #if hasattr(shared, 'train_in_ent_mean'):
                #    entropies[:, j] = \
                #        (entropies[:, j] - shared.train_in_ent_mean[j]) / \
                #        (shared.train_in_ent_std[j] + 1e-5)
                confidence[:, j] = ret_vals.confidence
                agreement[:, j] = ret_vals.agreement
                pred_labels[:, j] = ret_vals.pred_labels

            # Note, the returned labels are relative to the output head for CL1
            # and CL3 without split-heads. If we have a multi-head setting and
            # no information about the task-id, we need absolute labels!
            if config.cl_scenario == 1 or \
                    (config.cl_scenario == 3 and config.split_head_cl3 and config.fixed_embedding == -1):
                # Note, assuming all tasks have the same output head size.
                label_offset = i * data.num_classes
                labels += label_offset
                # Note, we also have to adapt the predicted labels accordingly,
                # as done below.

            # We choose the task embedding that leads to the predictive
            # distribution with lowest entropy.
            inferred_task_ids_ent = entropies.argmin(axis=1)
            # What if we were to infer task identity based on highest
            # confidence?
            inferred_task_ids_conf = confidence.argmax(axis=1)
            # What if we were choosing task identity according to where models
            # agree the most. Note, lower agreement score is better!
            inferred_task_ids_agree = agreement.argmin(axis=1)

            ### Compute average in- and out-of-distribution uncertainty.
            in_ents[i] = entropies[:, i].mean()
            in_conf = confidence[:, i].mean()
            in_agree = agreement[:, i].mean()
            if verbose:
                logger.debug('Test: Task %d - Mean entropy / confidence / ' % \
                             (i+1) + 'agreement on in-dist. samples: ' +
                             '%.2f / %.2f / %.2f.' \
                             % (in_ents[i], in_conf, in_agree))

            writer.add_scalar('test/task_%d/in_ents' % i, in_ents[i], n)
            writer.add_histogram('test/task_%d/in_ents_hist' % i,
                                 entropies[:, i], n)

            writer.add_scalar('test/task_%d/in_conf' % i, in_conf, n)
            writer.add_histogram('test/task_%d/in_conf_hist' % i,
                                 confidence[:, i], n)

            writer.add_scalar('test/task_%d/in_agree' % i, in_agree, n)
            writer.add_histogram('test/task_%d/in_agree_hist' % i,
                                 agreement[:, i], n)

            if n > 1:
                out_ents[i] = np.delete(entropies, optimal_embedding_task, axis=1).mean()
                out_conf = np.delete(confidence, optimal_embedding_task, axis=1).mean()
                out_agree = np.delete(agreement, optimal_embedding_task, axis=1).mean()
                if verbose:
                    logger.debug('Test: Task %d - Mean entropy / confidence ' %\
                                 (i+1)+'/ agreement on out-of-dist. samples: ' +
                                 '%.2f / %.2f / %.2f.' \
                                 % (out_ents[i], out_conf, out_agree))

                writer.add_scalar('test/task_%d/out_ents' % i, out_ents[i], n)
                writer.add_histogram('test/task_%d/out_ents_hist' % i,
                                     np.delete(entropies, i, axis=1), n)

                writer.add_scalar('test/task_%d/out_conf' % i, out_conf, n)
                writer.add_histogram('test/task_%d/out_conf_hist' % i,
                                     np.delete(confidence, i, axis=1), n)

                writer.add_scalar('test/task_%d/out_agree' % i, out_agree, n)
                writer.add_histogram('test/task_%d/out_agree_hist' % i,
                                     np.delete(agreement, i, axis=1), n)

            if i == n-1:
                # We want to produce one tensorboard plot that nicely summarizes
                # the in-distribution entropies of all tasks. So we always only
                # add the entropy values of the currently trained task.
                writer.add_scalar('test/during_in_ents', in_ents[i], n)
                writer.add_histogram('test/during_in_ents_hist',
                                     entropies[:, i], n)
                try:
                    # Allows a better visualization of the differences between
                    # tasks.
                    writer.add_histogram('test/during_log_in_ents_hist',
                                         np.log(entropies[:, i]), n)
                except:
                    if verbose:
                        logger.warn('Could not write log-entropy histogram ' +
                                    'for task %d.' % n)

            ### Compute task accuracy if task identitiy is inferred from entropy
            # Note, for CL1 this is actually not necessary to be computed, but
            # still interesting to report.
            pred_labels_inferred_ent = pred_labels[np.arange(labels.size),
                                                   inferred_task_ids_ent]
            pred_labels_inferred_conf = pred_labels[np.arange(labels.size),
                                                    inferred_task_ids_conf]
            pred_labels_inferred_agree = pred_labels[np.arange(labels.size),
                                                     inferred_task_ids_agree]
            # Important, for CL1 and CL3 with split-heads, we need to change
            # these labels to absulute labels, as they are currently measured
            # relative to the output head.
            if config.cl_scenario == 1 or \
                    (config.cl_scenario == 3 and config.split_head_cl3):
                # Note, assuming all tasks have the same output head size.
                pred_labels_inferred_ent += inferred_task_ids_ent * \
                    data.num_classes
                pred_labels_inferred_conf += inferred_task_ids_conf * \
                    data.num_classes
                pred_labels_inferred_agree += inferred_task_ids_agree * \
                    data.num_classes

            num_correct_ent = np.sum(pred_labels_inferred_ent == labels)
            num_correct_conf = np.sum(pred_labels_inferred_conf == labels)
            num_correct_agree = np.sum(pred_labels_inferred_agree == labels)
            acc_task_inferred_ent[i] = 100. * num_correct_ent / \
                data.num_test_samples
            acc_task_inferred_conf[i] = 100. * num_correct_conf / \
                data.num_test_samples
            acc_task_inferred_agree[i] = 100. * num_correct_agree / \
                data.num_test_samples
            
            # confusion matrices - TAg
            shared.summary['confusion_matrix_acc_tag'][n_testids-1, i] = acc_task_inferred_ent[i]
            if i < n_testids-1:
                shared.summary['confusion_matrix_forg_tag'][n_testids-1, i] = shared.summary['confusion_matrix_acc_tag'][:n_testids-1, i].max(0) - shared.summary['confusion_matrix_acc_tag'][n_testids-1, i]

            if verbose:
                logger.debug('Test: Task %d - Accuracy when using inferred ' % \
                             (i+1) + 'embeddings based on entropy / ' +
                             'confidence / agreement: %.2f%% / %.2f%% / %.2f%%.' \
                                % (acc_task_inferred_ent[i],
                                   acc_task_inferred_conf[i],
                                   acc_task_inferred_agree[i]))
            writer.add_scalar('test/task_%d/acc_task_inferred_ent' % i,
                              acc_task_inferred_ent[i], n)
            writer.add_scalar('test/task_%d/acc_task_inferred_conf' % i,
                              acc_task_inferred_conf[i], n)
            writer.add_scalar('test/task_%d/acc_task_inferred_agree' % i,
                              acc_task_inferred_agree[i], n)

            ### Compute accuracy of task inference
            # I.e., how often the correct tast embedding would have been chosen.
            # Note, this is again not necessary for CL1, but interesting.
            num_correct_ent = np.sum(inferred_task_ids_ent == i)
            num_correct_conf = np.sum(inferred_task_ids_conf == i)
            num_correct_agree = np.sum(inferred_task_ids_agree == i)
            task_inference_acc_ent[i] = 100. * num_correct_ent / \
                data.num_test_samples
            task_inference_acc_conf[i] = 100. * num_correct_conf / \
                data.num_test_samples
            task_inference_acc_agree[i] = 100. * num_correct_agree / \
                data.num_test_samples

            if verbose:
                logger.debug('Test: Task %d - Task inference accuracy based ' %\
                             (i+1) + 'on entropy / confidence / agreement: ' +
                             '%.2f%% / %.2f%% / %.2f%%.' \
                             % (task_inference_acc_ent[i],
                                task_inference_acc_conf[i],
                                task_inference_acc_agree[i]))
            writer.add_scalar('test/task_%d/task_inference_acc_ent' % i,
                              task_inference_acc_ent[i], n)
            writer.add_scalar('test/task_%d/task_inference_acc_conf' % i,
                              task_inference_acc_conf[i], n)
            writer.add_scalar('test/task_%d/task_inference_acc_agree' % i,
                              task_inference_acc_agree[i], n)

            ### Assess forgetting.
            # Note, only if a hyper-hypernet is present, it makes sense to
            # quantify forgetting by looking at how the output of the hyper-
            # hypernet `w_theta` has changed.
            if hasattr(shared, 'during_weights') and \
                    shared.during_weights is not None:
                W_curr = torch.cat([d.clone().view(-1) \
                                    for d in current_weights])
                # Put old models on CPU, to not run out of GPU memory when
                # training long task sequences.
                W_curr = W_curr.detach().cpu()
                if type(shared.during_weights[i]) == int:
                    assert(shared.during_weights[i] == -1)
                    shared.during_weights[i] = W_curr
                else:
                    W_during = shared.during_weights[i]
                    W_dis = torch.norm(W_curr - W_during, 2)
                    logger.debug('Test: Task %d - Euclidean distance ' % (i+1) +
                                 'to original hypernet output: %g' % W_dis)
                    writer.add_scalar('test/task_%d/hnet_out_forgetting' % i,
                                      W_dis, n)

    ### Compute task-inference accuracy superposing the task embeddings.
    if hasattr(config, 'supsup_task_inference') and \
            config.supsup_task_inference:
        assert np.all(np.equal(range(n), test_ids))

        if config.train_from_scratch:
            raise NotImplementedError()

        ### For each data set (i.e., for each task).
        task_inference_acc_ent_grad = []
        for i in test_ids:
            data = data_handlers[i]

            ### Task inference.
            # As opposed to above, here we generate a superposed model, which
            # consists of a weighted sum of task-specific embeddings, in order
            # to obtain a weight-generator that represents a combination of
            # all tasks, and generate predictions using this superposed model.
            # Then we compute the gradient of the entropy with respect to the
            # weight given to each model (alpha loadings), and select the task
            # as the one having the largest negative gradient (i.e. the task
            # whose contribution to the entropy computation can lead to the
            # largest decrease).

            # The alphas are initialized uniformly across models.
            alphas = 1/shared.num_trained * torch.ones((shared.num_trained),
                requires_grad=True, device=device)

            for _ in range(config.supsup_grad_steps):
                # Compute task-inference accuracy on data from task i.
                if method == 'avb' or method == 'ssge':
                    raise NotImplementedError('TODO')
                else:
                    disable_lrt = config.disable_lrt_test if \
                        hasattr(config, 'disable_lrt_test') else False
                    acc, ret_vals = pmutils.compute_acc(i, data, mnet,
                        hnet, device, config, shared, split_type='test',
                        return_dataset=False, return_entropies=True,
                        deterministic_sampling=True, alphas=alphas,
                        disable_lrt=disable_lrt, normal_post=normal_post, logger=logger)
                entropy = ret_vals.entropies

                # We choose the task embedding leading to the highest decrease
                # in entropy grad (argmax -dH = argmin dH).
                # FIXME. Incredibly inefficient, how to do this in a one-liner?
                # ent_grad = torch.autograd.grad(entropy.sum(), alphas, 
                #   only_inputs=True)[0]
                inferred_task_ids_ent = np.empty(len(entropy))
                all_ent_grad = []
                for b in range(len(entropy)): # iterate across batch size
                    retain_graph = True if b < len(entropy)-1 else False
                    # Compute the gradient of the entropy for the specific input
                    # with respect to the alpha values.
                    ent_grad = torch.autograd.grad(entropy[b], \
                        alphas, only_inputs=True, retain_graph=retain_graph)[0]
                    all_ent_grad.append(ent_grad.detach().cpu().numpy())
                    inferred_task_ids_ent[b] = torch.argmin(ent_grad).item()
                    assert inferred_task_ids_ent[b] in range(shared.num_trained)

                # Compute average gradient across all input samples.
                alphas_grad = torch.tensor(all_ent_grad, device=device)\
                    .mean(axis=0)

                # Update the alphas (only useful for more than one iteration).
                alphas = alphas.detach().clone() - config.supsup_lr*alphas_grad
                alphas.requires_grad = True

            ### Compute accuracy of task inference
            # I.e., how often the correct tast embedding would have been chosen.
            # Note, this is again not necessary for CL1, but interesting.
            supsup_num_correct_ent = np.sum(inferred_task_ids_ent == i)
            task_inference_acc_ent_grad.append(100.* supsup_num_correct_ent /\
                data.num_test_samples)

            if verbose:
                logger.debug('Test: Task %d - Task inference accuracy based ' %\
                             (i+1) + 'on entropy gradient: %.2f%%.' \
                             % (task_inference_acc_ent_grad[i]))
            writer.add_scalar('test/task_%d/task_inference_acc_ent_grad' % i,
                              task_inference_acc_ent_grad[i], n)

    ### Update performance summary.
    s = shared.summary

    s['acc_task_given'][:n] = acc_task_given
    s['acc_task_given_during'][n-1] = acc_task_given[n-1]
    s['acc_task_inferred_ent'][:n] = acc_task_inferred_ent
    s['acc_task_inferred_ent_during'][n-1] = acc_task_inferred_ent[n-1]
    if fixed_embedding != -1:
        s['acc_task_fixemb'][n-1] = acc_task_fixemb

    s['acc_avg_task_given'] = acc_task_given.mean() #if tested_all else -1
    s['acc_avg_task_given_during'] = np.mean(s['acc_task_given_during'][:n]) #if tested_all else -1
    s['acc_avg_task_inferred_ent'] = acc_task_inferred_ent.mean() #if tested_all else -1
    s['acc_avg_task_inferred_ent_during'] = \
        np.mean(s['acc_task_inferred_ent_during'][:n]) #if tested_all else -1
    if fixed_embedding != -1:
        s['avg_acc_task_fixemb'] = np.mean(acc_task_fixemb[:n])
    
    s['avg_task_inference_acc_ent'] = task_inference_acc_ent.mean()  \
        #if tested_all else -1
    s['acc_avg_task_inferred_conf'] = acc_task_inferred_conf.mean()  \
        #if tested_all else -1
    s['avg_task_inference_acc_conf'] = task_inference_acc_conf.mean()  \
        #if tested_all else -1
    s['acc_avg_task_inferred_agree'] = acc_task_inferred_agree.mean()  \
        #if tested_all else -1
    s['avg_task_inference_acc_agree'] = task_inference_acc_agree.mean()  \
        #if tested_all else -1

    #s['avg_acc_incr_given'] = np.mean(s['acc_task_given_during'][:n])
    #s['avg_acc_incr_inferred_ent'] = np.mean(s['acc_task_inferred_ent_during'][:n])

    if hasattr(config, 'supsup_task_inference') and \
            config.supsup_task_inference:
        s['avg_task_inference_acc_ent_grad'] = \
            np.mean(task_inference_acc_ent_grad) if tested_all else -1
        s['task_inference_acc_ent_grad'] = task_inference_acc_ent_grad

    if config.cl_scenario == 1:
        # Accuracy has been computed assuming we always knew the correct
        # task embedding.
        s['acc_avg_during'] = np.mean(s['acc_task_given_during'][:n])
        s['acc_avg_final'] = s['acc_avg_task_given']
    else:
        # Task identity is unknown. Hence, we need to infer the
        # embedding to use in order to compute an accuracy.
        # FIXME Is entropy really the best measure?
        # Note, during accuracies don't make a lot of sense, since they depend
        # on the number of tasks seen so far.
        s['acc_avg_during'] = np.mean(s['acc_task_inferred_ent_during'][:n])
        s['acc_avg_final'] = s['acc_avg_task_inferred_ent']

    pmutils.save_summary_dict(config, shared, shared.experiment_type)

    # If we haven't tested on all tasks, then we can't state average
    # performance.
    #if not tested_all:
    #    return

    ### Log test results.
    logger.debug('Test: Avg. in-dist. entropy %.2f (std: %.2f).'
                % (in_ents.mean(), in_ents.std()))
    writer.add_scalar('test/in_ents', in_ents.mean(), n)

    if n > 1:
        logger.debug('Test: Avg. out-of-dist. entropy %.2f (std: %.2f).'
                    % (out_ents.mean(), out_ents.std()))
        writer.add_scalar('test/out_ents', out_ents.mean(), n)

    if fixed_embedding != -1:
        logger.debug('Test: Avg. entropy using fixed %.2f (std: %.2f).'
                    % (ent_task_fixemb.mean(), ent_task_fixemb.std()))
        writer.add_scalar('test/ent_mean_fixemb', ent_task_fixemb.mean(), n)
        writer.add_scalar('test/ent_std_fixemb', ent_task_fixemb.std(), n)
        writer.add_histogram('test/ent_fixemb', ent_task_fixemb, n)

    logger.info('Test: Avg. task inference accuracy (entropy): ' +
                '%.2f (std: %.2f).' % (task_inference_acc_ent.mean(),
                                       task_inference_acc_ent.std()))
    writer.add_scalar('test/task_inference_acc_ent',
                      task_inference_acc_ent.mean(), n)
    logger.info('Test: Avg. task inference accuracy (confidence): ' +
                '%.2f (std: %.2f).' % (task_inference_acc_conf.mean(),
                                       task_inference_acc_conf.std()))
    writer.add_scalar('test/task_inference_acc_conf',
                      task_inference_acc_conf.mean(), n)
    logger.info('Test: Avg. task inference accuracy (agreement): ' +
                '%.2f (std: %.2f).' % (task_inference_acc_agree.mean(),
                                       task_inference_acc_agree.std()))
    writer.add_scalar('test/task_inference_acc_agree',
                      task_inference_acc_agree.mean(), n)
    
    if hasattr(config, 'supsup_task_inference') and \
            config.supsup_task_inference:
        logger.info('Test: Avg. task inference accuracy (entropy-grad): ' +
                    '%.2f (std: %.2f).' % (np.mean(task_inference_acc_ent_grad),
                                           np.std(task_inference_acc_ent_grad)))
        writer.add_scalar('test/task_inference_acc_ent_grad',
                          np.mean(task_inference_acc_ent_grad), n)

    logger.info('Test: Avg. accuracy when using correct embeddings: ' +
                '%.2f (std: %.2f).'
                % (acc_task_given.mean(), acc_task_given.std()))
    writer.add_scalar('test/acc_task_given', acc_task_given.mean(), n)

    logger.info('Test: Avg. accuracy when using inferred embeddings ' +
                 '(entropy): %.2f (std: %.2f).'
                 % (acc_task_inferred_ent.mean(), acc_task_inferred_ent.std()))
    writer.add_scalar('test/acc_task_inferred_ent',
                      acc_task_inferred_ent.mean(), n)
    logger.info('Test: Avg. accuracy when using inferred embeddings ' +
                 '(confidence): %.2f (std: %.2f).'
                 % (acc_task_inferred_conf.mean(),
                    acc_task_inferred_conf.std()))
    writer.add_scalar('test/acc_task_inferred_conf',
                      acc_task_inferred_conf.mean(), n)
    logger.info('Test: Avg. accuracy when using inferred embeddings ' +
                 '(agreement): %.2f (std: %.2f).'
                 % (acc_task_inferred_agree.mean(),
                    acc_task_inferred_agree.std()))
    writer.add_scalar('test/acc_task_inferred_agree',
                      acc_task_inferred_agree.mean(), n)
    if fixed_embedding != -1:
        logger.info('Test: Avg. accuracy when using fixed embeddings ' +
                    '%.2f (std: %.2f).' % (acc_task_fixemb.mean(), acc_task_fixemb.std()))
        writer.add_scalar('test/acc_task_fixemb',
                        acc_task_fixemb.mean(), n)
    logger.info('Test: Incremental Avg. accuracy (up to now) correct embeddings: '+
                '%.2f.' % s['acc_avg_task_given_during'])
    writer.add_scalar('test/acc_incr_task_given',
                      s['acc_avg_task_given_during'], n)
    logger.info('Test: Incremental Avg. accuracy (up to now) inferred embeddings: '+
                '%.2f.' % s['acc_avg_task_given_during'])
    writer.add_scalar('test/acc_avg_incr_task_inferred',
                      s['acc_avg_task_inferred_ent_during'], n)

    logger.info('Test: CL%d accuracy: %.2f%% (no. of tasks: %d).'
                % (config.cl_scenario, s['acc_avg_final'], n))
    writer.add_scalar('test/cl%d_accuracy' % config.cl_scenario,
                      s['acc_avg_final'], n)

    logger.info('Test: Confusion matrices. [verbose ON] TAw. Acc. Forg. TAg. Acc. Forg.')
    if verbose:
        logger.debug('TAw. Acc. \n' + np.array2string(s['confusion_matrix_acc_tag'], precision=2, separator=','))
        logger.debug('TAw. Forg. \n' + np.array2string(s['confusion_matrix_forg_tag'], precision=2, separator=','))
        logger.debug('TAg. Acc. \n' + np.array2string(s['confusion_matrix_acc_taw'], precision=2, separator=','))
        logger.debug('TAg. Forg. \n' + np.array2string(s['confusion_matrix_forg_taw'], precision=2, separator=','))
        
    sns.heatmap(s['confusion_matrix_acc_tag'], annot=True, cmap='Blues', cbar=False, vmin=0, vmax=100)
    writer.add_figure('test/TAg_acc', plt.gcf(), n)
    sns.heatmap(s['confusion_matrix_forg_tag'], annot=True, cmap='Blues', cbar=False, vmin=0, vmax=100)
    writer.add_figure('test/TAg_forg', plt.gcf(), n)
    sns.heatmap(s['confusion_matrix_acc_taw'], annot=True, cmap='Blues', cbar=False, vmin=0, vmax=100)
    writer.add_figure('test/TAw_acc', plt.gcf(), n)
    sns.heatmap(s['confusion_matrix_forg_taw'], annot=True, cmap='Blues', cbar=False, vmin=0, vmax=100)
    writer.add_figure('test/TAw_forg', plt.gcf(), n)

    if not tested_all:
        return
    logger.info('### Testing all trained tasks ... Done ###')

def evaluate(task_id, data, mnet, hnet, hhnet, dis, device, config, shared,
             logger, writer, train_iter, which_cond_id=-1, projection=True):
    """Evaluate the network(s) on the current task.

    Evaluate the performance of  the network on a single task on the
    validation set.

    Note:
         If no validation set is available, the test set will be used instead.

    Args:
         (....): See docstring of function :func:`train`.
        train_iter: The current training iteration.
    """

    if which_cond_id == -1:
        which_cond_id = task_id

    logger.info('# Evaluating network on task %d ' % (task_id + 1) +
                'before running training step %d ...' % (train_iter))

    pcutils.set_train_mode(False, mnet, hnet, hhnet, dis)

    # Note, this function is called during training, where the temperature is
    # not changed.
    assert shared.softmax_temp[task_id] == 1.

    with torch.no_grad():
        split_name = 'test' if data.num_val_samples == 0 else 'validation'
        if split_name == 'test':
            logger.debug('Eval - Using test set as no validation set is ' +
                         'available.')

        # In contrast, we visualize uncertainty using the test set.
        acc, ret_vals = pcutils.compute_acc(task_id, data, mnet, hnet, hhnet,
            device, config, shared, split_type='val', return_entropies=True,
            return_samples=True, which_cond_id=which_cond_id, logger=logger, projection=True)

        logger.info('Eval - Accuracy on %s set: %f%%.' % (split_name, acc))
        writer.add_scalar('eval/task_%d/accuracy' % task_id, acc,
                          train_iter)

        if not np.isnan(ret_vals.entropies.mean()):
            logger.debug('Eval - Entropy on %s set: %f (std: %f).'
                         % (split_name, ret_vals.entropies.mean(),
                            ret_vals.entropies.std()))
            writer.add_scalar('eval/task_%d/mean_entropy' % task_id,
                              ret_vals.entropies.mean(), train_iter)
            writer.add_histogram('eval/task_%d/entropies' % task_id,
                                 ret_vals.entropies, train_iter)
        else:
            logger.warning('NaN entropy has been detected during evaluation!')

        ### Compute discriminator accuracy.
        if dis is not None and hnet is not None:
            hnet_theta = None
            if hhnet is not None:
                hnet_theta = hhnet.forward(cond_id=which_cond_id)

            # FIXME Is it ok if I only look at how samples from the current
            # implicit distribution are classified?
            dis_out, dis_inputs = pcutils.process_dis_batch(config, shared,
                config.val_sample_size, device, dis, hnet, hnet_theta,
                dist=None)
            dis_acc = (dis_out > 0).sum() / config.val_sample_size * 100.
            dis_acc = dis_acc.detach().cpu().numpy()

            logger.debug('Eval - Discriminator accuracy: %.2f%%.' % (dis_acc))
            writer.add_scalar('eval/task_%d/dis_acc' % task_id, dis_acc,
                              train_iter)

            # FIXME Summary results should be written in the test method after
            # training on a task has finished (note, eval is no guaranteed to be
            # called after or even during training). But I just want to get an
            # overview.
            s = shared.summary
            s['acc_dis'][task_id] = dis_acc
            s['acc_avg_dis'] = np.mean(s['acc_dis'][:(task_id+1)])

            # Visualize weight samples.
            # FIXME A bit hacky.
            w_samples = dis_inputs
            if config.use_batchstats:
                w_samples = dis_inputs[:, (dis_inputs.shape[1]//2):]
            pcutils.visualize_implicit_dist(config, task_id, writer, train_iter,
                                            w_samples, figsize=(10, 6))

        elif 'ssge' in shared.experiment_type and hnet is not None:
            hnet_theta = None
            if hhnet is not None:
                hnet_theta = hhnet.forward(cond_id=which_cond_id)
            w_samples = ssgeu.generate_weight_sample(config, shared,
                device, hnet, hnet_theta, num_samples=config.val_sample_size,
                ret_format='flattened')
            pcutils.visualize_implicit_dist(config, task_id, writer, train_iter,
                                            w_samples, figsize=(10, 6))

        logger.info('# Evaluating training ... Done')

def train(task_id, data, mnet, hnet, hhnet, dis, device, config, shared, logger,
          writer, method='avb', task_id_emb=-1, projection=True):
    r"""Train a Bayesian Neural Network continually using either a prior-focused
    training approach or a posterior replay approach.

    In both cases, we use variational inference and utilize a hypernetwork
    ``hnet`` to model an implicit distribution over main network parameters.

    When training the implicit hypernetwork ``hnet`` using prior-focused CL
    (determined via attribute ``shared.prior_focused``), forgetting will be
    mitigated through the prior matching (i.e., if ``task_id > 0`` we checkpoint
    the hypernet ``hnet`` at the beginning of training to use its samples as
    prior samples). The loss function in this case will be

    .. math::
        
        KL\big(q_\theta(W) \mid\mid q_{\theta^*}(W)  \big) - \
        \mathbb{E}_{q_\theta(W)} \big[ \log p(\mathcal{D} \mid W) \big]

    where :math:`q_{\theta^*}(W)` denotes the implicit distribution realized
    through the hypernet ``hnet`` before training on the current task's data
    :math:`\mathcal{D}`, which is denoted by the checkpointed weights
    :math:`\theta^*`.

    In case a task-specific posterior should be trained, then the prior
    :math:`q_{\theta^*}(W)` in the above equation is replaced by an arbitrary
    prior :math:`p(W)` and the ``hnet`` weights become task-dependent
    :math:`\theta^{(t)}` due to the use of the hyper-hypernetwork ``hhnet``.

    Forgetting is prevented due to the simple output preserving regularizer
    proposed `here <https://arxiv.org/abs/1906.00695>`__ applied to the
    hyper-hypernet.

    Thus, the total loss will be the task-specific variational inference (VI)
    loss plus the CL regularizer.

    .. math::
        \text{loss} = \text{task\_vi\_loss} + \beta * \text{cl\_regularizer}

    In case coresets are used to regularize towards high entropy on previous
    tasks (in case of a prior-focused method, this can only make sense in a
    multi-head setting), the loss becomes

    .. math::
        \text{loss} = \text{task\_vi\_loss} + \beta * \text{regularizer} + \
            \gamma * \text{coreset\_regularizer}

    In order to add constraint on the last layer between the sampled 
    weight networks, we add off-diagonal penalty - the loss becomes
    .. math::
        \text{loss} = \text{task\_vi\_loss} + \beta * \text{regularizer} + \
            \gamma * \text{coreset\_regularizer} + \delta * \text{oof\_diag\_simW}

    In order to add constraint on the fixed embedding, similar to prior-based methods
        we are able to force the posterior distributions to match their samples,
        for instance using CW metric.

    Since the prior-matching involves implicit distributions, it has to be
    approximated. We use either the `AVB <https://arxiv.org/abs/1701.04722>`__
    algorithm for this purpose, where the discriminator ``dis`` is trained to
    estimate the log-density ratio, or
    `SSGE <https://arxiv.org/abs/1806.02925>`__ that directly estimates the
    gradients of the log density of our implicit distribution.

    Args:
        task_id: The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        hnet: The model of the hyper network (may be ``None``).
        hhnet: The hyper-hypernetwork (may be ``None``).
        dis: The discriminator (may be ``None``). If not discriminator is
            present, the prior-matching doesn't enter the loss (the
            task-specific becomes a maximum-likelihood loss).
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared: Miscellaneous data shared among training functions.
        logger: Command-line logger.
        writer: The tensorboard summary writer.
        method (str, optional): The algorithm used to train the networks. 
            Possible values are: ``avb`` and ``ssge``. Note that if ``ssge``
            option is selected, the discriminator provided is supposed to be
            ``None``.
        task_id_emb: Which embedding will be used for the training
        projection: Whether to use nested latent samples
    """
    if method != 'avb':
        assert dis is None

    detach_grad_emb = False

    if task_id_emb == -1:
        task_id_emb = task_id
    else:
        detach_grad_emb = True

    logger.info('Training network on task %d using %s...' % (task_id+1, method))
    logger.info(f'Will use {task_id_emb} embedding')
    if projection:
        #v = torch.normal(torch.zeros(shared.noise_dim), config.latent_std).to(device)
        #v = v.detach()
        #shared.projection_v.append(v.detach())
        v = shared.projection_v[task_id]
        logger.info(f'Projecting the latent Z on less dim space. Init {v}')

    pcutils.set_train_mode(True, mnet, hnet, hhnet, dis)

    # Whether we train a classification or regression task?
    is_regression = 'regression' in shared.experiment_type
    if is_regression:
        eval_func = regavb.evaluate

        assert config.ll_dist_std > 0
        ll_scale = 1. / config.ll_dist_std**2
    else:
        eval_func = evaluate

        # Note, during training we start with a temperature of 1 (i.e., we can
        # ignore it when computing the network output (e.g., to evaluate the
        # cross-entropy loss)).
        # Only after training (of all tasks), the temperature may be changed for
        # calibration reasons.
        assert shared.softmax_temp[task_id] == 1.

    # Which outputs should we consider from the main network for the current
    # task.
    allowed_outputs = pmutils.out_units_of_task(config, data, task_id,
                                                task_id+1)

    # It might be that tasks are very similar and we can transfer knowledge
    # from the previous solution.
    if hhnet is not None and config.init_with_prev_emb and task_id > 0:
        # All conditional parameters (usually just task embeddings) are task-
        # specific and used for reinitialization.
        last_emb = get_conditional_parameters(hhnet, task_id-1)
        for ii, cemb in enumerate(get_conditional_parameters(hhnet, task_id)):
            cemb.data = last_emb[ii].data

    ####################
    ### Define Prior ###
    ####################
    # Should we even do prior-matching?
    perform_pm = hnet is not None and (dis is not None or method != 'avb') \
        and (config.kl_scale != 0 or config.kl_schedule != 0)
    if not perform_pm:
        logger.warn('No prior-matching will be performed for task %d.' \
                    % (task_id+1))

    # If yes, can we use the adaptive contrast trick?
    # Note, if the prior distribution is implicit, we can't use the trick.
    use_adaptive_contrast = method == 'avb' and perform_pm and \
        not config.no_adaptive_contrast and \
        (shared.prior_focused and task_id == 0 or \
         not shared.prior_focused and not config.use_prev_post_as_prior)

    if use_adaptive_contrast:
        logger.debug('Adaptive contrast trick is used to stabilize prior-' +
                     'matching for task %d.' % (task_id+1))

    if perform_pm:
        prior_mean, prior_std, prior_dist, prior_theta = None, None, None, None

        use_explicit_prior = task_id == 0 or (not shared.prior_focused and \
            not config.use_prev_post_as_prior)
        if use_explicit_prior:
            prior_mean = torch.cat([p.flatten() for p in shared.prior_mean])
            prior_std = torch.cat([p.flatten() for p in shared.prior_std])

            prior_dist = Normal(prior_mean, prior_std)

            logger.debug('An explicit Gaussian prior is used for VI.')

        else:
            if config.train_from_scratch:
                # FIXME Doesn't make sense to use randomly initialized hnet
                # as prior. Instead, we should reload a checkpoint of the
                # previous network.
                raise NotImplementedError()

            logger.debug('Checkpointing implicit hypernet to use it as ' +
                         'prior ...')
            # FIXME This is not sufficient if the hypernet uses tricks such as
            # batchnorm. Instead, we should create an actual model checkpoint on
            # disk and have a second `hnet` instance!
            prior_theta = [p.detach().clone() \
                           for p in hnet.unconditional_params]

            if allowed_outputs is not None:
                # FIXME Head-specific weights should be set to the Gaussian
                # prior, as we haven't learned a posterior for them when
                # learning the previous task.
                logger.warn('Prior for head-specific weights not set ' +
                            'correctly (implementation missing).')

    # Plot prior predictive distribution.
    if perform_pm and shared.experiment_type.startswith('gmm'):
        pgutils.plot_gmm_prior_preds(task_id, data, mnet, hnet, hhnet, device,
            config, shared, logger, writer,
            shared.prior_mean if prior_theta is None else None,
            shared.prior_std if prior_theta is None else None,
            prior_theta=prior_theta, which_cond_id=task_id_emb)

    ############################
    ### Setup CL regularizer ###
    ############################
    # Note, the CL regularizer is only used if we train a posterior per task
    # using the `hhnet`.
    # FIXME We don't have to regularize all weights of the hypernetwork `hnet`
    # with the CL regularizer, as some of them might only be connected to output
    # heads of the main network that are not trained on the current data.

    # Whether the regularizer will be computed during training?
    calc_reg = hhnet is not None  and task_id > 0 and config.beta > 0 and \
        not config.train_from_scratch

    # Regularizer targets.
    if calc_reg:
        if config.calc_hnet_reg_targets_online:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            # We need to checkpoint the current hyper-hypernet. Note, this will
            # be the only memory overhead and doesn't grow with the number of
            # tasks.
            targets_hhnet = None
            prev_hhnet_theta = [p.detach().clone() \
                                for p in hhnet.unconditional_params]
            prev_task_embs = [p.detach().clone() \
                              for p in hhnet.conditional_params]
        else:
            targets_hhnet = hreg.get_current_targets(task_id, hhnet, shared.prev_task_id)
            prev_hhnet_theta = None
            prev_task_embs = None

    ###########################
    ### Create optimizer(s) ###
    ###########################
    assert mnet is not None
    assert hnet is not None or hhnet is None

    if hnet is None:
        params = mnet.parameters()
        assert len(list(mnet.parameters())) == len(mnet.weights)
    elif hhnet is None:
        params = hnet.parameters()
        assert len(list(hnet.parameters())) == len(hnet.unconditional_params)
    else:
        params = hhnet.parameters()
        assert len(list(hhnet.parameters())) == len(hhnet.internal_params)

    #if projection:
    #    params += [v]

    hoptimizer = tutils.get_optimizer(params, config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop, use_adadelta=config.use_adadelta,
        use_adagrad=config.use_adagrad)

    # Discriminator optimizer.
    doptimizer = None
    # Note, we don't need a discriminator that estimate the log-density
    # ratio, if we don't perform prior-matching.
    if method == 'avb' and perform_pm:
        assert dis is not None

        dis_lr = config.lr if config.dis_lr == -1 else config.dis_lr
        doptimizer = tutils.get_optimizer(dis.parameters(), dis_lr,
            momentum=config.momentum, weight_decay=config.weight_decay,
            use_adam=config.use_adam, adam_beta1=config.adam_beta1,
            use_rmsprop=config.use_rmsprop, use_adadelta=config.use_adadelta,
            use_adagrad=config.use_adagrad)

    ################################
    ### Learning rate schedulers ###
    ################################
    # FIXME We apply the schedulers currently not to the discriminator training.

    plateau_scheduler = None
    lambda_scheduler = None
    if config.plateau_lr_scheduler:
        assert config.epochs != -1
        # The scheduler config has been taken from here:
        # https://keras.io/examples/cifar10_resnet/
        # Note, we use 'max' instead of 'min' as we look at accuracy rather
        # than validation loss in classification!
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau( \
            hoptimizer, 'min' if is_regression else 'max', factor=np.sqrt(0.1),
            patience=5, min_lr=0.5e-6, cooldown=0)

    if config.lambda_lr_scheduler:
        assert config.epochs != -1

        lambda_scheduler = optim.lr_scheduler.LambdaLR(hoptimizer,
            tutils.lambda_lr_schedule)

    ######################
    ### Start training ###
    ######################
    # FIXME If the hypernetwork were to use batchnorm, we would also need to
    # make sure that we supply the correct batchnorm stats if it is learned
    # continually via a hyper-hypernetwork!
    mnet_kwargs = pmutils.mnet_kwargs(config, task_id, mnet)

    num_train_iter, iter_per_epoch = sutils.calc_train_iter( \
        data.num_train_samples, config.batch_size, num_iter=config.n_iter,
        epochs=config.epochs)

    for i in range(num_train_iter):
        #########################
        ### Evaluate networks ###
        #########################
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            eval_func(task_id, data, mnet, hnet, hhnet, dis, device, config,
                      shared, logger, writer, i, which_cond_id=task_id_emb)
            pcutils.set_train_mode(True, mnet, hnet, hhnet, dis)

        if i % 100 == 0:
            logger.debug('Training iteration: %d.' % i)

        #####################################
        ### Adaptive Contrast Preparation ###
        #####################################
        ac_mean, ac_std, ac_dist = None, None, None
        if use_adaptive_contrast:
            assert hnet is not None
            # We need to compute the mean and diagonal covariance matrix of
            # our current variational approximation represented by the `hnet`.
            ac_mean, ac_std = pcutils.estimate_implicit_moments(config, shared,
                task_id_emb, hnet, hhnet, config.num_ac_samples, device)

            ac_dist = Normal(ac_mean, ac_std)

        ###########################
        ### Train Discriminator ###
        ###########################
        loss_dis = 0
        acc_dis = 0
        if perform_pm and method == 'avb':
            assert hnet is not None

            real_dist = ac_dist if use_adaptive_contrast else prior_dist
            real_theta = prior_theta
            assert (use_explicit_prior and real_dist is not None) or \
                   (not use_explicit_prior and real_theta is not None)

            # What is the current implicit distribution?
            fake_theta = None
            if hhnet is not None:
                fake_theta = hhnet.forward(cond_id=task_id_emb)
                # Note, when training the discriminator, we don't want to
                # backprop through the `hhnet`.
                fake_theta = [p.detach() for p in fake_theta]

            dis_targets = torch.cat([torch.zeros(config.dis_batch_size),
                torch.ones(config.dis_batch_size)]).to(device)


            for i_dis in range(config.num_dis_steps):
                doptimizer.zero_grad()

                real, _ = pcutils.process_dis_batch(config, shared,
                    config.dis_batch_size, device, dis, hnet, real_theta,
                    dist=real_dist)
                # Process samples from the current implicit distribution.
                fake, _ = pcutils.process_dis_batch(config, shared,
                    config.dis_batch_size, device, dis, hnet, fake_theta,
                    dist=None)

                # Note, this loss applies the sigmoid to the discriminator
                # outputs.
                loss_dis = F.binary_cross_entropy_with_logits( \
                    torch.cat([real, fake], dim=0).squeeze(), dis_targets)

                # Compute discriminator accuracy.
                if i_dis == config.num_dis_steps - 1:
                    # Note, prior samples (real) are correctly "classified" if
                    # the output of the discriminator is negative.
                    # The opposite is true for samples from the implicit
                    # hypernet (fake).
                    acc_dis = gan.accuracy(fake, real, 0) * 100.0

                loss_dis.backward()
                doptimizer.step()

        ################################
        ### Train Implicit Posterior ###
        ################################
        hoptimizer.zero_grad()

        # What is the current implicit distribution?
        theta_current = None
        if hhnet is not None:
            theta_current = hhnet.forward(cond_id=task_id_emb)


        ### Prior-matching term.
        w_samples = None
        loss_kl = 0
        loss_prior_ssge = 0
        loss_cw = 0
        loss_mmd = 0
        loss_ot = 0
        if perform_pm:
            if method == 'avb':
                # If the sample sizes for both MC estimates are the same, then
                # we reuse the weight samples already found.
                loss_kl, w_samples = pcutils.calc_prior_matching(config, shared,
                    config.num_kl_samples, device, dis, hnet, theta_current,
                    prior_dist, ac_dist,
                    return_current_samples=config.num_kl_samples == \
                                           config.train_sample_size)
            elif method == 'ssge':
                # Compute the prior term.
                if use_explicit_prior:
                    loss_prior_ssge = ssgeu.get_prior_loss_term(config, shared,
                        config.num_kl_samples, device, hnet, theta_current,
                        prior_dist)
                else:
                    # If we are using a prior-focused approach, the gradient of
                    # the loss prior term also needs to be estimated via SSGE.
                    grad_prior_ssge = ssgeu.get_prior_grad_term(config, shared,
                        device, logger, config.num_kl_samples, hnet,
                        theta_current, prior_theta)

                # Compute entropy term.
                # Note, everything within this function, that is used to
                # estimate the gradient directly, should occur in a separate
                # graph and not interfere with the autograd gradient computation
                # of the loss later on (when `backward` is called).
                theta_current_ssge_grad = ssgeu.estimate_entropy_grad(config,
                    shared, device, logger, hnet, theta_current)

                if config.gamma_cw != 0 and task_id > 0:
                    # dists = torch.empty(config.train_sample_size)
                    generator = None
                    #if deterministic_sampling:
                    #    generator = torch.Generator()#device=device)
                    #    # Note, PyTorch recommends using large random seeds:
                    #    # https://tinyurl.com/yx7fwrry
                    #    generator.manual_seed(2147483647)
                    # for ii in range(config.train_sample_size):
                    ns = config.train_sample_size
                    
                    for ii in range(ns):
                        rand_z = torch.normal(torch.zeros(1, shared.noise_dim),
                            config.latent_std, generator=generator).to(device)

                        prev = hnet.forward(uncond_input=rand_z, weights=shared.theta_prev)
                        curr = hnet.forward(uncond_input=rand_z, weights=theta_current)
                            
                        prev_1d = torch.cat([p.detach().flatten() for p in prev])
                        curr_1d = torch.cat([p.flatten() for p in curr])

                        if ii == 0:
                            prev_sample = torch.empty((ns, len(prev_1d)))
                            curr_sample = torch.empty((ns, len(curr_1d)))
                        prev_sample[ii] = prev_1d
                        curr_sample[ii] = curr_1d

                    optimal_gamma = ssgeu.silverman_rule_of_thumb_sample(torch.cat((prev_sample, curr_sample), 0))
                    loss_cw = ssgeu.cw(torch.Tensor(prev_sample), torch.Tensor(curr_sample), optimal_gamma)

                if config.gamma_mmd != 0 and task_id > 0:
                    # dists = torch.empty(config.train_sample_size)
                    generator = None
                    #if deterministic_sampling:
                    #    generator = torch.Generator()#device=device)
                    #    # Note, PyTorch recommends using large random seeds:
                    #    # https://tinyurl.com/yx7fwrry
                    #    generator.manual_seed(2147483647)
                    # for ii in range(config.train_sample_size):
                    ns = config.train_sample_size
                    
                    for ii in range(ns):
                        rand_z = torch.normal(torch.zeros(1, shared.noise_dim),
                            config.latent_std, generator=generator).to(device)

                        prev = hnet.forward(uncond_input=rand_z, weights=shared.theta_prev)
                        prev_1d = torch.cat([p.detach().flatten() for p in prev])
                        curr = hnet.forward(uncond_input=rand_z, weights=theta_current)
                        curr_1d = torch.cat([p.flatten() for p in curr])

                        if ii == 0:
                            prev_sample = torch.empty((ns, len(prev_1d)))
                            curr_sample = torch.empty((ns, len(curr_1d)))
                        prev_sample[ii] = prev_1d
                        curr_sample[ii] = curr_1d

                    loss_mmd = ssgeu.mmd(torch.Tensor(prev_sample), torch.Tensor(curr_sample))

                if config.gamma_ot != 0 and task_id > 0:
                    # dists = torch.empty(config.train_sample_size)
                    generator = None
                    #if deterministic_sampling:
                    #    generator = torch.Generator()#device=device)
                    #    # Note, PyTorch recommends using large random seeds:
                    #    # https://tinyurl.com/yx7fwrry
                    #    generator.manual_seed(2147483647)
                    # for ii in range(config.train_sample_size):
                    ns = config.train_sample_size
                    
                    for ii in range(ns):
                        rand_z = torch.normal(torch.zeros(1, shared.noise_dim),
                            config.latent_std, generator=generator).to(device)

                        prev = hnet.forward(uncond_input=rand_z, weights=shared.theta_prev)
                        prev_1d = torch.cat([p.detach().flatten() for p in prev])
                        curr = hnet.forward(uncond_input=rand_z, weights=theta_current)
                        curr_1d = torch.cat([p.flatten() for p in curr])

                        if ii == 0:
                            prev_sample = torch.empty((ns, len(prev_1d)))
                            curr_sample = torch.empty((ns, len(curr_1d)))
                        prev_sample[ii] = prev_1d
                        curr_sample[ii] = curr_1d

                    loss_ot = ssgeu.sink_stabilized(torch.Tensor(prev_sample), torch.Tensor(curr_sample), reg=1)

        ### Compute negative log-likelihood (NLL).
        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        if not is_regression:
            # Modify 1-hot encodings according to CL scenario.
            assert(T.shape[1] == data.num_classes)
            # In CL1, CL2 and CL3 (with seperate heads) we do not have to modify
            # the targets.
            if config.cl_scenario == 3 and (config.fixed_embedding != -1 or not config.split_head_cl3) and \
                    task_id > 0:
                # We prepend zeros to the 1-hot vector according to the number
                # of output units belonging to previous tasks.
                T = torch.cat((torch.zeros((config.batch_size,
                    task_id * data.num_classes)).to(device), T), dim=1)

            _, labels = torch.max(T, 1) # Integer labels.
            labels = labels.detach()

        loss_nll = 0
        mean_train_acc = 0
        loss_sim_weight = 0
        prev_weights = []

        for j in range(config.train_sample_size):
            if w_samples is not None:
                weights = w_samples[j]
            elif hnet is not None:
                z = torch.normal(torch.zeros(1, shared.noise_dim),
                                 config.latent_std).to(device)
                if projection:
                    z = z - ((z @ v) * (1/torch.norm(v)**2)).reshape(-1,1) @ v.reshape((1,-1))
                weights = hnet.forward(uncond_input=z, weights=theta_current)
            else:
                weights = None

            prev_weights.append(weights)
            for prev in range(j):
                loss_sim_weight += pcutils.compute_weight_sim(weights, prev_weights[prev])

            Y = mnet.forward(X, weights=weights, **mnet_kwargs)
            if allowed_outputs is not None:
                Y = Y[:, allowed_outputs]

            # Task-specific loss.
            # We use the reduction method 'mean' on purpose and scale with
            # the number of training samples below.
            if is_regression:
                loss_nll += 0.5 * ll_scale * F.mse_loss(Y, T, reduction='mean')
            else:
                # Note, that `cross_entropy` also computed the softmax for us.
                loss_nll += F.cross_entropy(Y, labels, reduction='mean')

                # Compute accuracy on batch.
                # Note, softmax wouldn't change the argmax.
                _, pred_labels = torch.max(Y, 1)
                mean_train_acc += 100. * torch.sum(pred_labels == labels) / \
                    config.batch_size

        loss_nll *= data.num_train_samples / config.train_sample_size
        # Note, the mean accuracy is different from the accuracy of the
        # predictive distribution (where the softmax outputs are meaned before
        # computing an accuracy), which is done in the `compute_acc` function.
        mean_train_acc /= config.train_sample_size

        ### Compute CL regularizer.
        loss_reg = 0
        if calc_reg:
            loss_reg = hreg.calc_fix_target_reg(hhnet, task_id,
                targets=targets_hhnet, prev_theta=prev_hhnet_theta,
                prev_task_embs=prev_task_embs,
                batch_size=config.hnet_reg_batch_size)

        ### Compute coreset regularizer.
        cs_reg = 0
        if config.coreset_size != -1 and \
                (task_id > 0 or config.past_and_future_coresets):
            assert hasattr(shared, 'coreset')
            if config.past_and_future_coresets:
                cs_all = shared.coreset[shared.task_ident != task_id]
                batch_inds = np.random.randint(0, cs_all.shape[0],
                                               config.coreset_batch_size)
                coreset = cs_all[batch_inds]
            else:
                batch_inds = np.random.randint(0, shared.coreset.shape[0],
                                               config.coreset_batch_size)
                coreset = shared.coreset[batch_inds]
            #cs_task_ids = shared.task_ident[batch_inds]
            #cs_tasks = np.unique(cs_task_ids)

            if is_regression:
                # FIXME How to push uncertainty on previous tasks?
                raise NotImplementedError('Coresets not yet implemented for ' +
                                          'regression!')

            # Construct maximum entropy targets for coreset samples.
            target_size = len(allowed_outputs) if config.cl_scenario != 2 \
                    else data.num_classes
            cs_targets = torch.ones(config.coreset_batch_size, target_size). \
                to(device) / target_size

            for j in range(config.train_sample_size):
                weights = None
                if hnet is not None:
                    z = torch.normal(torch.zeros(1, shared.noise_dim),
                                     config.latent_std).to(device)
                    weights = hnet.forward(uncond_input=z,
                                           weights=theta_current)

                cs_preds = mnet.forward(coreset, weights=weights, **mnet_kwargs)
                if allowed_outputs is not None:
                    cs_preds = cs_preds[:, allowed_outputs]

                cs_reg += Classifier.softmax_and_cross_entropy(cs_preds,
                                                               cs_targets)
                #cs_reg += Classifier.knowledge_distillation_loss(cs_preds,
                #                                                 cs_targets)

            cs_reg /= config.train_sample_size

        kl_scale = pmutils.calc_kl_scale(config, num_train_iter, i, logger)

        loss = kl_scale * (loss_kl + loss_prior_ssge) + loss_nll + \
            config.beta * loss_reg + config.coreset_reg * cs_reg + \
            config.gamma * loss_sim_weight + \
            config.gamma_cw * loss_cw + config.gamma_mmd * loss_mmd + config.gamma_ot * loss_ot

        # We may need to compute further derivatives later on, that's why in
        # some cases we retain the graph.
        loss.backward(retain_graph=True \
            if method == 'ssge' and hhnet is not None else False)
        if config.clip_grad_value != -1:
            torch.nn.utils.clip_grad_value_( \
                hoptimizer.param_groups[0]['params'], config.clip_grad_value)
        elif config.clip_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(hoptimizer.param_groups[0]['params'],
                                           config.clip_grad_norm)
        if method == 'ssge':
            # When using SSGE, the estimated gradient of the entropy term of
            # the ELBO needs to be added.
            if hhnet is None:
                # If the theta parameters are contained within the hypernetwork,
                # add the computed gradients directly.
                for ii, theta_ii in enumerate(hnet.internal_params):
                    theta_ii.grad += kl_scale * theta_current_ssge_grad[ii]

                    # Add the gradient computated for the cross-entropy term of 
                    # the ELBO whenver using the previous posterior as prior.
                    if not use_explicit_prior:
                        theta_ii.grad += kl_scale * grad_prior_ssge[ii]
            else:
                # If the thetas are generated by a hyper-hypernetwork, the
                # computed gradient `theta_curr_grad_flat` has to be used to
                # compute the gradient of the thetas with respect to the
                # hyperhypernet parameters.
                theta_curr_flat = torch.cat([p.flatten() for p in
                                             theta_current])
                theta_curr_grad_flat = torch.cat([p.flatten() for p in
                                                  theta_current_ssge_grad])
                if not use_explicit_prior:
                    theta_curr_grad_flat += torch.cat([p.flatten() for p in
                                                       grad_prior_ssge])
                theta_curr_grad_flat *= kl_scale
                psi_current = list(hhnet.parameters())

                # Find latest element in `psi_current` that has a gradient.
                last_grad_ind = -1
                for ii in range(len(psi_current)-1, -1, -1):
                    if psi_current[ii].grad is not None:
                        last_grad_ind = ii
                        break
                assert last_grad_ind != -1

                for ii, psi_ii in enumerate(psi_current):
                    # Note, future task embeddings have not been used in the
                    # computation but are part of the hhnet parameters.
                    if psi_ii.grad is None: # Skip embeddings.
                        #print(hhnet.param_shapes_meta[ii])
                        continue
                    retain_graph = False if ii == last_grad_ind else True
                    # We have to set `allow_unused` to accomodate for the fact
                    # that previous task embeddings have been used for the
                    # computation of `loss` but not `theta_curr_flat`.
                    ii_grad = grad(outputs=theta_curr_flat, inputs=psi_ii,
                        grad_outputs=theta_curr_grad_flat,
                        retain_graph=retain_graph, allow_unused=True)[0]
                    if ii_grad is not None:
                        psi_ii.grad += ii_grad
                    #else:
                    #    print(hhnet.param_shapes_meta[ii])

        hoptimizer.step()

        ###############################
        ### Learning rate scheduler ###
        ###############################
        pmutils.apply_lr_schedulers(config, shared, logger, task_id, data, mnet,
            hnet, device, i, iter_per_epoch, plateau_scheduler,
            lambda_scheduler, hhnet=hhnet, method='avb')

        ###########################
        ### Tensorboard summary ###
        ###########################
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/kl_scale' % task_id, kl_scale, i)
            writer.add_scalar('train/task_%d/loss_kl_scaled' % task_id, kl_scale * (loss_kl + loss_prior_ssge), i)
            writer.add_scalar('train/task_%d/loss_sim' % task_id, loss_sim_weight, i)
            writer.add_scalar('train/task_%d/loss_sim_scaled' % task_id, config.gamma * loss_sim_weight, i)
            writer.add_scalar('train/task_%d/loss_kl' % task_id, (loss_kl + loss_prior_ssge), i)
            writer.add_scalar('train/task_%d/loss_nll' % task_id, loss_nll, i)
            writer.add_scalar('train/task_%d/regularizer' % task_id, loss_reg, i)
            writer.add_scalar('train/task_%d/regularizer_scaled' % task_id, config.beta * loss_reg, i)
            writer.add_scalar('train/task_%d/coreset_reg' % task_id, cs_reg, i)
            writer.add_scalar('train/task_%d/loss_cw' % task_id, loss_cw, i)
            writer.add_scalar('train/task_%d/loss_cw_scaled' % task_id, config.gamma_cw * loss_cw, i)
            writer.add_scalar('train/task_%d/loss_mmd' % task_id, loss_mmd, i)
            writer.add_scalar('train/task_%d/loss_mmd_scaled' % task_id, config.gamma_mmd * loss_mmd, i)
            writer.add_scalar('train/task_%d/loss_ot' % task_id, loss_ot, i)
            writer.add_scalar('train/task_%d/loss_ot_scaled' % task_id, config.gamma_ot * loss_ot, i)
            writer.add_scalar('train/task_%d/loss' % task_id, loss, i)
            if not is_regression:
                writer.add_scalar('train/task_%d/accuracy' % task_id,
                                  mean_train_acc, i)

            if method == 'avb':
                writer.add_scalar('train/task_%d/dis_loss' % task_id, \
                    loss_dis, i)
                writer.add_scalar('train/task_%d/dis_acc' % task_id, acc_dis, i)

            if ac_dist is not None:
                try:
                    writer.add_histogram('train/task_%d/estimated_mean' % \
                                         task_id, ac_mean, i)
                    writer.add_histogram('train/task_%d/estimated_std' % \
                                         task_id, ac_std, i)
                except:
                    logger.error('Couldn\'t write histograms of mean and ' +
                                 'std of current implicit distribution.')
                    raise ValueError('NaN in hypernet output!')

    pmutils.checkpoint_bn_stats(config, task_id, mnet)

    ### Update or setup the coresets (if requested by user).
    if config.coreset_size != -1 and not config.past_and_future_coresets:
        pmutils.update_coreset(config, shared, task_id, data, mnet, hnet,
            device, logger, allowed_outputs, hhnet=hhnet, method='avb')
    
    shared.theta_prev = [t.detach() for t in theta_current] # be careful, not detached!
    
    shared.prev_task_id.append(task_id_emb)
    if not shared.prior_focused:
        logger.debug(f'Corresponding embedding for task %d was {hhnet.conditional_params[task_id_emb].detach().cpu().numpy()}' % (task_id+1))

    #if task_id > 0:
    #    shared.projection_v[-1] = v.detach()
    #    logger.debug(f'Corresponding v vector for task %d was {v.detach().cpu().numpy()}' % (task_id+1))
    
    logger.info('Training network on task %d ... Done' % (task_id+1))

if __name__ == '__main__':
    # Consult README file!
    raise Exception('Script is not executable!')

