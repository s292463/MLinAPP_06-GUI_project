	�;�,z@�;�,z@!�;�,z@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:�;�,z@%#gaO;�?1y �H�vw@I_(`;�E@rEagerKernelExecute 0*	V-��u@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�� Z+�?!7Q��O@)��-�v��?1���O@:Preprocessing2F
Iterator::Modely"��p�?!}�%�K�9@)}�H�F��?19/ف-�,@:Preprocessing2U
Iterator::Model::ParallelMapV2uu�b�T�?!·r`j�&@)uu�b�T�?1·r`j�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?�ܵ��?! ���R@).�v����?1�p��@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatI���Σ�?!���ד�@)��vL݅?1���:�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�Qd���~?!Hg��M@)�Qd���~?1Hg��M@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�wE��?!j<BP@) p��s�j?1��y���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor@j'�;d?!�S����?)@j'�;d?1�S����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceJ��	�y[?!������?)J��	�y[?1������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�L<���$@Qmv�� iV@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	%#gaO;�?%#gaO;�?!%#gaO;�?      ��!       "	y �H�vw@y �H�vw@!y �H�vw@*      ��!       2      ��!       :	_(`;�E@_(`;�E@!_(`;�E@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�L<���$@ymv�� iV@