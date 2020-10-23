# QVAE
Official Pytorch implementation of quaternion-valued variational autoencoder (QVAE).

To run QVAE training, download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), install `requirements.txt` and type:
```
python train_qvae.py
```
Once trained the model, to generate new samples and reconstructions from the test set, type:
```
python generation.py
```
specify `--QVAE=True` to generate from QVAE.

Quaternion convolutions are borrowed from [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) by Titouan Parcollet.
