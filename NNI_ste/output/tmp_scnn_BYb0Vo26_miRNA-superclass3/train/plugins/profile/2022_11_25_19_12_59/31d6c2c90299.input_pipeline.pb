	2����3@2����3@!2����3@	�<eaߺ@�<eaߺ@!�<eaߺ@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02����3@��?�V?1�g����@I{�ᯡ*@Y`�n���?r0*	Zd;�D�@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapY���.4@!m��n�qO@)I�p@1&$�uO@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�M�q@!�݈�j@@)4����@1�P�TPD@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip:�۠�@!�L��3;B@)� �X4��?1/��^��	@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat%���?!�Q����?)h��n�?1�U֞���?:Preprocessing2F
Iterator::Modelu><K��?!�rn$���?)�Բ��H�?1����+��?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat*X�l:�?!�M ���?)�A���?1����5��?:Preprocessing2U
Iterator::Model::ParallelMapV2���aڗ?!WQO�ڙ�?)���aڗ?1WQO�ڙ�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�u�ݑ��?!��M��?)�u�ݑ��?1��M��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor2r���?!��&o�?)2r���?1��&o�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSlice[|
��z?!0�����?)[|
��z?10�����?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range5��-</u?!���p�K�?)5��-</u?1���p�K�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice[�[!��b?!����W7�?)[�[!��b?1����W7�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�66.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�<eaߺ@I20Ll��P@Q��uvb�9@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��?�V?��?�V?!��?�V?      ��!       "	�g����@�g����@!�g����@*      ��!       2      ��!       :	{�ᯡ*@{�ᯡ*@!{�ᯡ*@B      ��!       J	`�n���?`�n���?!`�n���?R      ��!       Z	`�n���?`�n���?!`�n���?b      ��!       JGPUY�<eaߺ@b q20Ll��P@y��uvb�9@