import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ReadmissionCNN import *

class utility:

    @staticmethod
    def parse_time(time_in_seconds):
        #print(f'time in seconds: {time_in_seconds}')
        if time_in_seconds < 60:
            return f'{round(time_in_seconds,2)} sec'
        elif time_in_seconds >= 60 and time_in_seconds < 3600:
            return f'{round(time_in_seconds/60,2)} min'
        elif time_in_seconds >= 3600 and time_in_seconds < 86400:
            return f'{round(time_in_seconds/60/60,2)} hr'
        else:
            return f'{round(time_in_seconds/60/60/24,2)} day'
    @staticmethod
    def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
        # TODO: Make plots for loss curves and accuracy curves.
        # TODO: You do not have to return the plots.
        # TODO: You can save plots as files by codes here or an interactive way according to your preference.

        # Plot training and validation loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        #plt.savefig('./output/train_valid_loss_plot.png')
        plt.show()
        plt.close()  # Close the current plot

        # Plot training and validation accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(valid_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        #plt.savefig('./output/train_valid_accuracy_plot.png')
        plt.show()
        plt.close()  # Close the current plot

    @staticmethod
    def plot_confusion_matrix(results, class_names):
        # TODO: Make a confusion matrix plot.
        # TODO: You do not have to return the plots.
        # TODO: You can save plots as files by codes here or an interactive way according to your preference.
        y_true, y_pred = zip(*results)
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Plot confusion matrix using seaborn
        plt.figure(figsize=(6, 4))
        sns.set(font_scale=1.2)  # for label size
        sns.heatmap(cm_normalized, annot=True, annot_kws={"size": 12}, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.legend()
        #plt.savefig('./output/confusion_plot.png')
        plt.show()
        plt.close()

    @staticmethod
    def compute_batch_accuracy(output, target):
        """Computes the accuracy for a batch"""
        with torch.no_grad():

            batch_size = target.size(0)
            _, pred = output.max(1)
            correct = pred.eq(target).sum()

            return correct * 100.0 / batch_size

    # @staticmethod
    # def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     losses = AverageMeter()
    #     accuracy = AverageMeter()

    #     model.train()

    #     end = time.time()
    #     for i, (input_data, target) in enumerate(data_loader):
    #         # measure data loading time
    #         data_time.update(time.time() - end)

    #         input_data = input_data.float()

    #         if isinstance(input_data, tuple):
    #             input_data = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input_data])
    #         else:
    #             input_data = input_data.to(device)

    #         target = target.to(device)

    #         optimizer.zero_grad()
    #         output = model(input_data)
    #         loss = criterion(output, target)
    #         assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

    #         loss.backward()
    #         optimizer.step()

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         losses.update(loss.item(), target.size(0))
    #         accuracy.update(utility.compute_batch_accuracy(output, target).item(), target.size(0))

    #         if i % print_freq == 0:
    #             print('Epoch: [{0}][{1}/{2}]\t'
    #                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
    #                 epoch, i, len(data_loader), batch_time=batch_time,
    #                 data_time=data_time, loss=losses, acc=accuracy))

    #     return losses.avg, accuracy.avg

    # @staticmethod
    # def evaluate(model, device, data_loader, criterion, print_freq=10):
    #     batch_time = AverageMeter()
    #     losses = AverageMeter()
    #     accuracy = AverageMeter()

    #     results = []
    #     correct_neg_indices = []
    #     correct_pos_indices = []

    #     model.eval()

    #     with torch.no_grad():
    #         end = time.time()
    #         for i, (input_data, target) in enumerate(data_loader):

    #             input_data = input_data.float()
    #             if isinstance(input_data, tuple):
    #                 input_data = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input_data])
    #             else:
    #                 input_data = input_data.to(device)

    #             target = target.to(device)
    #             #print(f'target:{target}')
    #             #print(f'input shape: {input[0].shape}')

    #             output = model(input_data)
    #             #print(f'output: {output}')
    #             loss = criterion(output, target)

    #             # measure elapsed time
    #             batch_time.update(time.time() - end)
    #             end = time.time()

    #             losses.update(loss.item(), target.size(0))
    #             accuracy.update(utility.compute_batch_accuracy(output, target).item(), target.size(0))

    #             y_true = target.detach().to('cpu').numpy().tolist()
    #             y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
    #             results.extend(list(zip(y_true, y_pred)))

    #             if i % print_freq == 0:
    #                 print('Test: [{0}/{1}]\t'
    #                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                     'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
    #                     i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

    #     for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    #         if true == pred:
    #             if true == 0:
    #                 correct_neg_indices.append(i)
    #             else:
    #                 correct_pos_indices.append(i)
    #     return losses.avg, accuracy.avg, results, (correct_neg_indices, correct_pos_indices)
    
    @staticmethod
    def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        model.train()

        end = time.time()
        for i, (input_data, target, original_batch_indices) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input_data = input_data.float()

            if isinstance(input_data, tuple):
                input_data = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input_data])
            else:
                input_data = input_data.to(device)

            target = target.to(device)

            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(utility.compute_batch_accuracy(output, target).item(), target.size(0))

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=accuracy))

        return losses.avg, accuracy.avg

    @staticmethod
    def evaluate(model, device, data_loader, criterion, print_freq=5):
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        results = []
        correct_neg_indices = []
        correct_pos_indices = []

        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input_data, target, original_batch_indices) in enumerate(data_loader):
                #print(f"Test Iteration Number: {i}")
                input_data = input_data.float()
                if isinstance(input_data, tuple):
                    input_data = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input_data])
                else:
                    input_data = input_data.to(device)

                target = target.to(device)
                #print(f'target:{target}')
                #print(f'input shape: {input_data.shape}')

                output = model(input_data)
                #print(f'output: {output}')
                loss = criterion(output, target)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                losses.update(loss.item(), target.size(0))
                accuracy.update(utility.compute_batch_accuracy(output, target).item(), target.size(0))

                y_true = target.detach().to('cpu').numpy().tolist()
                y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
                results.extend(list(zip(y_true, y_pred)))
                
                for _, (truth, pred, original_index) in enumerate(zip(y_true, y_pred,original_batch_indices)):
                    if truth == pred:
                        if truth == 0:
                            correct_neg_indices.append(original_index)
                        else:
                            correct_pos_indices.append(original_index)

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

        
        #print(y_pred)

        return losses.avg, accuracy.avg, results, correct_pos_indices, correct_neg_indices    

    @staticmethod
    def preprocess_sentence(regex_tokenizer, punctuation, stop_words , text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()        

        tokens = [token for token in regex_tokenizer.tokenize(text) 
                if token not in punctuation and 
                token not in stop_words
                ]

        processed_text = ' '.join(tokens)
        
        # processed_text = ' '.join(tokens).replace('unit numeric identifier', '') \
        #                                  .replace('admission date', '') \
        #                                  .replace('discharge date', '')\
        #                                  .replace('date birth', '')

        return processed_text.strip()

    @staticmethod
    def create_collate_fn(embedding_model, notes_len_cutoff):

        def custom_collate_fn(batch):
            
            def get_embedding_vector(word, model):
                # This function returns the embedding vector for a word from the pre-trained model
                try:
                    return model[word]
                except KeyError:
                    # If the word is not in the model, return a zero vector
                    return np.zeros(model.vector_size)

            #print(embedding_model.vector_size)

            batch_tokenized_notes, labels, original_batch_indices = zip(*batch)
            #print(batch_tokenized_notes)
            # Apply embeddings to the entire batch of texts
            #batch_embeddings = embedding_model(list(texts))

            batch_embeddings = []

            # Iterate over each record in the DataFrame
            for tokenized_note in batch_tokenized_notes:
            
                # Initialize the embedding matrix for this record with zeros
                notes_embedding_matrix = np.zeros((notes_len_cutoff, embedding_model.vector_size))
                
                # Iterate over the tokenized words up to the fixed_length
                for i, word in enumerate(tokenized_note[:notes_len_cutoff]):
                    # Get the embedding vector for the word
                    notes_embedding_matrix[i] = get_embedding_vector(word, embedding_model)
                
                # Add the record's embedding matrix to the list
                batch_embeddings.append(notes_embedding_matrix)

            # Convert the list of embeddings into a 3D NumPy array
            batch_embeddings_stacked = np.stack(batch_embeddings)
            #print(batch_embeddings_stacked.shape)
            
            batch_embeddings_tensor = torch.from_numpy(batch_embeddings_stacked).unsqueeze(1)
            # Convert the batch of embeddings and labels to tensors
            #batch_embeddings_tensor = torch.stack(batch_embeddings)
            #print(batch_embeddings_tensor.shape)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            return batch_embeddings_tensor, labels_tensor, original_batch_indices

        return custom_collate_fn