	�ҥI� @�ҥI� @!�ҥI� @	YXo�@YXo�@!YXo�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�ҥI� @)�ahur�?1�8�d��@A�v�1<�?I'�����?Y�-�R�?rEagerKernelExecute 0*	!�rh�=f@2F
Iterator::Model�!�Q*�?!?��;G@)M.��:��?1&�Xn�QA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�a�[>��?!��<�+=@)���)��?1�1q,VC9@:Preprocessing2U
Iterator::Model::ParallelMapV2/�
Ҍ�?!b4���'@)/�
Ҍ�?1b4���'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��
~b�?!��j�%�J@)��}�u�?1��=Y�$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceCƣT��?!��9.�@)CƣT��?1��9.�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�)[$�?!�$4��*)@)�p�a�ƃ?1�y�H�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5�\��u|?!��\��=@)5�\��u|?1��\��=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���{��?!M}�ϛ�+@)���ӹ�d?1(����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 17.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�20.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ZXo�@I��ҪB@Q��m�6�M@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	)�ahur�?)�ahur�?!)�ahur�?      ��!       "	�8�d��@�8�d��@!�8�d��@*      ��!       2	�v�1<�?�v�1<�?!�v�1<�?:	'�����?'�����?!'�����?B      ��!       J	�-�R�?�-�R�?!�-�R�?R      ��!       Z	�-�R�?�-�R�?!�-�R�?b      ��!       JGPUYZXo�@b q��ҪB@y��m�6�M@