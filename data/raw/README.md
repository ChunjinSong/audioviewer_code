### Audio Dataset
We use [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) dataset (LDC93S1) to train the audio model. Registration is required to access the dataset. For UBC users, the dataset is accessible on the [Abacus Data Network](https://abacus-library-ubc-ca.ezproxy.library.ubc.ca/dataverse/abacus).

#### Pre-processing
To generate mel spectrograms of the TIMIT dataset, run
~~~
python gen_melspecs.py ./configs/config_rawdata.yaml
~~~

To generate datasets of mel spectrogram segments by time step, run:
~~~
python ./preprocess_audio.py $SPLIT
~~~
where `$SPLIT` can be `ALL`, `TEST`, `TRAIN`, `KALDI_DEV`, `KALDI_TEST` for different data split.

To generate datasets of mel spectrogram segments by phones, run:
~~~
python ./preprocess_audio_anno.py $SPLIT $ALIGN
~~~
where `$ALIGN` can be `start` or `center` to align the phones.

### Visual Dataset
The visual model is trainied on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and [MNIST](http://yann.lecun.com/exdb/mnist/) dataset for a visualization of human face and hand written number respectively. Both datasets are accessible with [`torchvision.datasets`](https://pytorch.org/docs/stable/torchvision/datasets.html).
