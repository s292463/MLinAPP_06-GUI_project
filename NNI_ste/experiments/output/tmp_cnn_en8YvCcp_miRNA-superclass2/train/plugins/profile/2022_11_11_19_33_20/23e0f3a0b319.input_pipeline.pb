	��]/M�@��]/M�@!��]/M�@	��$q��@��$q��@!��$q��@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��]/M�@��'�?1���6T, @AZh�4��?Ioc�#�7	@YA��Ljh�?rEagerKernelExecute 0*	�|?5^��@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap<��ؖ��?!I�:��O@) �C��<�?1�o/��CM@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map-|}�K��?!z���?@)��ϷK�?1��ĺ
�;@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat5�l�/�?!���'�@)X S�?114/�@:Preprocessing2F
Iterator::Model�we���?!",\!S�@).�l�IF�?1r��0,�@:Preprocessing2U
Iterator::Model::ParallelMapV2�$�9ϔ?!�E;$���?)�$�9ϔ?1�E;$���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�?�&M��?!A�2��?)��_>Y�?1�r:�J��?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�C���?!�V�A�?)���E�?1��$H��?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�*��p�?!o�|�w�?)�*��p�?1o�|�w�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice��?!�l��k�?)��?1�l��k�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��i�?!RJ�n!P@)'��>V�{?1����*@�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�"j��Gy?!�F�Ck�?)�"j��Gy?1�F�Ck�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�b�J!p?!ȳug:�?)�b�J!p?1ȳug:�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenategd��S�?!�ۊ%�?)7l[�� c?1^�9�$\�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor2t�R?!��lګ�?)2t�R?1��lګ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 22.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�44.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��$q��@IA}N91�P@Q��|>�><@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��'�?��'�?!��'�?      ��!       "	���6T, @���6T, @!���6T, @*      ��!       2	Zh�4��?Zh�4��?!Zh�4��?:	oc�#�7	@oc�#�7	@!oc�#�7	@B      ��!       J	A��Ljh�?A��Ljh�?!A��Ljh�?R      ��!       Z	A��Ljh�?A��Ljh�?!A��Ljh�?b      ��!       JGPUY��$q��@b qA}N91�P@y��|>�><@