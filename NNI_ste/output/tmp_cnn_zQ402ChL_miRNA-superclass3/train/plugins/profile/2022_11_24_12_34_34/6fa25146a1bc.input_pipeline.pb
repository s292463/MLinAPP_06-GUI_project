	/�o�S�y@/�o�S�y@!/�o�S�y@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:/�o�S�y@be4�y��?1g����)w@I�)V�`E@rEagerKernelExecute 0*	�p=
�f@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate����o�?!�LBo~E@)�JC�B�?1����G1D@:Preprocessing2F
Iterator::Model2s��cͰ?!8�R"��B@)q��H/j�?18�d���9@:Preprocessing2U
Iterator::Model::ParallelMapV2���0a�?!sz�PY�&@)���0a�?1sz�PY�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�XP�i�?!�a��_kO@)�el�f�?1��6��\!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�B �8�?!��f�Z�"@)2q� ��?1�؝�!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�'*�Tv?!x$��@)�'*�Tv?1x$��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����(@�?!)ư�dF@)�e��
j?1��=�m��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor_~�Ɍ�e?!�F��?)_~�Ɍ�e?1�F��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice���_?!8�����?)���_?18�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIo���$@Q��ΣgV@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	be4�y��?be4�y��?!be4�y��?      ��!       "	g����)w@g����)w@!g����)w@*      ��!       2      ��!       :	�)V�`E@�)V�`E@!�)V�`E@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qo���$@y��ΣgV@