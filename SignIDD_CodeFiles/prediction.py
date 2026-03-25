import torch
import numpy as np

from data import Dataset, make_data_iter
from helpers import calculate_dtw
from batch import Batch
from model import Model
from constants import PAD_TOKEN


# Validate epoch given a dataset
# def validate_on_data(model: Model,
#                      data: Dataset,
#                      batch_size: int,
#                      max_output_length: int,
#                      eval_metric: str,
#                      loss_function: torch.nn.Module = None,
#                      batch_type: str = "sentence",
#                      type = "val",
#                      BT_model = None):

#     valid_iter = make_data_iter(
#         dataset=data, batch_size=batch_size,
#         shuffle=True, train=False)

#     pad_index = model.src_vocab.stoi[PAD_TOKEN]
#     # disable dropout
#     model.eval()
#     # don't track gradients during validation
#     with torch.no_grad():
#         valid_hypotheses = []
#         valid_references = []
#         valid_inputs = []
#         file_paths = []
#         all_dtw_scores = []

#         valid_loss = 0
#         total_ntokens = 0
#         total_nseqs = 0

#         batches = 0
#         for valid_batch in iter(valid_iter):
#             # Extract batch
#             batch = Batch(torch_batch=valid_batch,
#                           pad_index=pad_index,
#                           model=model)
#             targets = batch.trg_input

#             # run as during training with teacher forcing
#             if loss_function is not None and batch.trg is not None:
#                 # Get the loss for this batch
#                 batch_loss = model.get_loss_for_batch(is_train=True,
#                                                          batch=batch,
#                                                          loss_function=loss_function)

#                 valid_loss += batch_loss
#                 total_ntokens += batch.ntokens
#                 total_nseqs += batch.nseqs

#             output = model.forward(src=batch.src,
#                                        trg_input=batch.trg_input[:, :, :150],
#                                        src_mask=batch.src_mask,
#                                        src_lengths=batch.src_lengths,
#                                        trg_mask=batch.trg_mask,
#                                        is_train=False)
            
#             output = torch.cat((output, batch.trg_input[:, :, 150:]), dim=-1)
            
#             # Add references, hypotheses and file paths to list
#             valid_references.extend(targets)
#             valid_hypotheses.extend(output)
#             file_paths.extend(batch.file_paths)
#             # Add the source sentences to list, by using the model source vocab and batch indices
#             valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
#                                  range(len(batch.src))])

#             # Calculate the full Dynamic Time Warping score - for evaluation
#             dtw_score = calculate_dtw(targets, output)
#             all_dtw_scores.extend(dtw_score)

#             # Can set to only run a few batches
#             # if batches == math.ceil(20/batch_size):
#             #     break
#             batches += 1

#         # Dynamic Time Warping scores
#         current_valid_score = np.mean(all_dtw_scores)

#     return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
#            valid_inputs, all_dtw_scores, file_paths
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     vocab=None,
                     # tok_fun=None,
                     trg_size: int = 150,
                     BT_model=None):

    # Build validation DataLoader
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        vocab=vocab,
        # tok_fun=tok_fun,
        trg_size=trg_size,
        shuffle=False
    )

    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    model.eval()

    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Wrap into Batch
            batch = Batch(torch_batch=valid_batch,
                          pad_index=pad_index,
                          model=model)
            targets = batch.trg_input

            # Compute loss if available
            if loss_function is not None and batch.trg is not None:
                batch_loss = model.get_loss_for_batch(
                    is_train=True,
                    batch=batch,
                    loss_function=loss_function
                )
                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # Forward pass
            output = model.forward(
                src=batch.src,
                trg_input=batch.trg_input[:, :, :trg_size],
                src_mask=batch.src_mask,
                src_lengths=batch.src_lengths,
                trg_mask=batch.trg_mask,
                is_train=False
            )

            # Concatenate remainder of target
            output = torch.cat((output, batch.trg_input[:, :, trg_size:]), dim=-1)

            # Collect results
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            valid_inputs.extend([
                [model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))]
                for i in range(len(batch.src))
            ])

            # Dynamic Time Warping score
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)

            batches += 1
            
            # Clean up memory periodically during validation
            if batches % 10 == 0:
                torch.cuda.empty_cache()

        current_valid_score = np.mean(all_dtw_scores)

    return (
        current_valid_score,
        valid_loss,
        valid_references,
        valid_hypotheses,
        valid_inputs,
        all_dtw_scores,
        file_paths
    )
