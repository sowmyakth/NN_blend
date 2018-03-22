""" Basic Script to run CNN deblender"""
from __future__ import division
import os
import basic_net_utils as utils
import numpy as np
import pickle


def load_data(filename):
    train_data = np.load(filename)
    X_train = train_data['X_train']
    Y_train = train_data['Y_train']
    X_val = train_data['X_val']
    Y_val = train_data['Y_val']
    return X_train, Y_train, X_val, Y_val


def main(Args):
    # run_ident = Args.name + str(Args.learn_rate)
    run_ident = Args.name + Args.loss_fn
    path = '/global/cscratch1/sd/sowmyak/training_data'
    filename = os.path.join(path, 'stamps.npz')
    X_train, Y_train, X_val, Y_val = load_data(filename)
    model = utils.CNN_deblender(config=True, num_cnn_layers=6,
                                run_ident=run_ident,
                                learning_rate=Args.learn_rate,
                                loss_fn=Args.loss_fn)
    run_params = utils.Meas_args(epochs=Args.epochs,
                                 batch_size=Args.batch_size,
                                 print_every=500)
    output = model.run_basic(X_train, Y_train,
                             run_params, X_val, Y_val)
    [train_loss, val_loss, pred, ind_loss] = output
    model.save()
    model.sess.close()
    data = {'X_val': X_val,
            'Y_val': Y_val,
            'pred': pred,
            'ind_loss': ind_loss,
            'train_loss': train_loss,
            'val_loss': val_loss}
    path = os.path.join(os.path.dirname(os.getcwd()), "outputs",
                        run_ident + '_data.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.sess.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', default=5e-4, type=float,
                        help="Learning rate of net [Default:5e-4]")
    parser.add_argument('--epochs', default=100, type=int,
                        help="Number of times net trained on entire training\
                        set [Default:100]")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Size of each mini batch [Default:32]")
    parser.add_argument('--name', default='lr_', type=str,
                        help="string to save model outputs [Default:lr_]")
    parser.add_argument('--loss_fn', choices=['l2', 'l1', 'chi_sq'],
                        help="string to save model outputs [Default:l2]",
                        default='l2')
    args = parser.parse_args()
    main(args)
