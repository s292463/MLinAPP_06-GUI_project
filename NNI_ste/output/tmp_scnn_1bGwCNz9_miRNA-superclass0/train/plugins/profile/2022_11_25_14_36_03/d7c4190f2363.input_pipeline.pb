	m��}D<@m��}D<@!m��}D<@	5��#)�@5��#)�@!5��#)�@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0m��}D<@���͋�?1��]���2@I�YO�N@Y�4�;� @r0*	�Zd���@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map�|^�#@!tɁf?V@)^d~��#@1��[�V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�! 86�?!�v��rf#@)�����?1YCf��"@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�]=�1�?!�;� A�?)3p@KW��?1���X��?:Preprocessing2F
Iterator::Model�X���?!r�V���?)��e�-�?1���%��?:Preprocessing2U
Iterator::Model::ParallelMapV2�L��ݤ?!�:�[��?)�L��ݤ?1�:�[��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���8�?!w�ކ��$@)�~j�t��?1�2�����?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatm�s�p�?!D�t����?)�u��S�?1���3K�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice]�gA(�?!�����w�?)]�gA(�?1�����w�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchg�lt�O�?!�!J��?)g�lt�O�?1�!J��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��Y�h9�?!N���~I�?)��Y�h9�?1N���~I�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�VBwI�u?!�B{ܰ[�?)�VBwI�u?1�B{ܰ[�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceD� ��c?!��]���?)D� ��c?1��]���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�25.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no94��#)�@I����L:@QH��E�P@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���͋�?���͋�?!���͋�?      ��!       "	��]���2@��]���2@!��]���2@*      ��!       2      ��!       :	�YO�N@�YO�N@!�YO�N@B      ��!       J	�4�;� @�4�;� @!�4�;� @R      ��!       Z	�4�;� @�4�;� @!�4�;� @b      ��!       JGPUY4��#)�@b q����L:@yH��E�P@