	n/�)@n/�)@!n/�)@	��y]�@��y]�@!��y]�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLn/�)@j0�G��?1_���:�!@A�s�v�4�?I�a����?YS���t�?rEagerKernelExecute 0*��n��f@)       =2F
Iterator::Model߈�Y�h�?!	� �G@)�5�o���?1_*چ�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\�O��?!��m�X?@)���{�?1��rk�y;@:Preprocessing2U
Iterator::Model::ParallelMapV2��ID��?!�{Rn2@)��ID��?1�{Rn2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����9�?!Qx�	X�/@)*:��H�?1��fF�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicer����)�?!�@���I@)r����)�?1�@���I@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZippUj��?!���]�J@)L�$zł?1oi{T=@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�q�j��|?!J! ��@)�q�j��|?1J! ��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps����(�?!�^���l1@)àL���h?1(b��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 13.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��y]�@I�����;@Q*
[.�vQ@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	j0�G��?j0�G��?!j0�G��?      ��!       "	_���:�!@_���:�!@!_���:�!@*      ��!       2	�s�v�4�?�s�v�4�?!�s�v�4�?:	�a����?�a����?!�a����?B      ��!       J	S���t�?S���t�?!S���t�?R      ��!       Z	S���t�?S���t�?!S���t�?b      ��!       JGPUY��y]�@b q�����;@y*
[.�vQ@