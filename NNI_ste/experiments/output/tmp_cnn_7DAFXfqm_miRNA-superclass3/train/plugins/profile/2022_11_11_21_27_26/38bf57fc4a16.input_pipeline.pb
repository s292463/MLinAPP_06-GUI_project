	���( @���( @!���( @	�#�q�@�#�q�@!�#�q�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���( @���g� �?1�y ��J@AR��񘁚?I���k���?Y�����?rEagerKernelExecute 0*	�&1�a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�#d Ϧ?!˲_6I@@)A�v�?1��(�D;@:Preprocessing2F
Iterator::Model��>rk�?!. K�/�B@)Z� ͠?1�����7@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�aQ��?!��h~,@)�aQ��?1��h~,@:Preprocessing2U
Iterator::Model::ParallelMapV2��}�<�?!�qr#�x+@)��}�<�?1�qr#�x+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���|�͵?!����"O@)=dʇ�j�?1Rv���'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�>:u峜?!|4(~~4@)9F�G��?1�'�N��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5&�\R�}?!^g�L6@)5&�\R�}?1^g�L6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\:�<c�?!z�UCi6@)�� ��ze?1��T��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�21.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�#�q�@I��IhC@Q�kɏx�M@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���g� �?���g� �?!���g� �?      ��!       "	�y ��J@�y ��J@!�y ��J@*      ��!       2	R��񘁚?R��񘁚?!R��񘁚?:	���k���?���k���?!���k���?B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?b      ��!       JGPUY�#�q�@b q��IhC@y�kɏx�M@