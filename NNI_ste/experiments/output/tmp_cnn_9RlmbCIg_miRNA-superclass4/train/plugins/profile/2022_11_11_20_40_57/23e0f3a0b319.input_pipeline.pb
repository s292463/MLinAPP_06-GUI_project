	����Z�@����Z�@!����Z�@	�8Li(d@�8Li(d@!�8Li(d@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL����Z�@kIG9��?137߈�y @A�V'g(�?I���~3�
@Y�zܷZ'�?rEagerKernelExecute 0*	f;�O��@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"�4��?!�2�
N@)��c���?1j� ��x?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�	MK�?!�s�q@w:@)���-�?1�犟4[:@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map^��yȔ�?!T�0��=@)�~O�S��?1w�s�7Z6@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatj��4ӽ�?!x��.��@)� ��%s�?1�ĵ��@:Preprocessing2F
Iterator::Model#/kb��?!]���sx@)���i�:�?1�Qk��@:Preprocessing2U
Iterator::Model::ParallelMapV2N^d~��?!�����9@)N^d~��?1�����9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��K��?!ߕTiy@)�"j��G�?1��8RΥ�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchs�,&6�?!X�@���?)s�,&6�?1X�@���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipl�f���?!J��뤋O@)t��gy|?1qX�M�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�D��)x?!�X���?)�D��)x?1�X���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice~T�~O�s?!��g�?)~T�~O�s?1��g�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeW��mUr?!��I�H&�?)W��mUr?1��I�H&�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�m�2{?!f<3��p�?)h^��^?1=�I'�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor ����M?!�Z��?) ����M?1�Z��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�8Li(d@Ir��Z�<P@Q��j{(�<@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	kIG9��?kIG9��?!kIG9��?      ��!       "	37߈�y @37߈�y @!37߈�y @*      ��!       2	�V'g(�?�V'g(�?!�V'g(�?:	���~3�
@���~3�
@!���~3�
@B      ��!       J	�zܷZ'�?�zܷZ'�?!�zܷZ'�?R      ��!       Z	�zܷZ'�?�zܷZ'�?!�zܷZ'�?b      ��!       JGPUY�8Li(d@b qr��Z�<P@y��j{(�<@