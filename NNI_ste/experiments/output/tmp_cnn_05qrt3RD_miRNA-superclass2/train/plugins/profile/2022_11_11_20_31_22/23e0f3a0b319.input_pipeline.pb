	�_YiRz%@�_YiRz%@!�_YiRz%@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC�_YiRz%@�>rk�-@1�v�1�@A��	�_�?IR~R��Q@rEagerKernelExecute 0*	6^�Ie�@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapHN&n��?!f��i�9P@)}]��t�?1���sM@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map!��	L��?!Q����l5@)�^Pj�?1��N4�.@:Preprocessing2F
Iterator::Model�CV�z�?!V���I%$@)��]ؚ��?1�~I��@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatg{��ȥ?!�/i�ܿ@)��h���?1���ކ@:Preprocessing2U
Iterator::Model::ParallelMapV2�
(��G�?!��醤�@)�
(��G�?1��醤�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�9Dܜ�?!��d�-@)s����?1��7�d�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat((E+��?!Ŭ��ҹ@)g���u�?1�YN$���?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch��w�?!R�$�f�?)��w�?1R�$�f�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipʇ�j�j�?!�_�� Q@)߿yq�}?1U��g,�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�Ŧ�B w?!2��6�?)�Ŧ�B w?12��6�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����#*t?!�}9Ϝ��?)����#*t?1�}9Ϝ��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeQ�\�mOp?!�}�+���?)Q�\�mOp?1�}�+���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�Xl���z?!9Zt�0�?)��M�qZ?1�킔��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorX��C�Q?!�_��<�?)X��C�Q?1�_��<�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 25.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�52.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�tiGQ�S@QQ,Z��5@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�>rk�-@�>rk�-@!�>rk�-@      ��!       "	�v�1�@�v�1�@!�v�1�@*      ��!       2	��	�_�?��	�_�?!��	�_�?:	R~R��Q@R~R��Q@!R~R��Q@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�tiGQ�S@yQ,Z��5@