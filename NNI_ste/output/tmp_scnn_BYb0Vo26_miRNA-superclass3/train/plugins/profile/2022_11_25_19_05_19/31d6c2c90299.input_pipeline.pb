	���$!2@���$!2@!���$!2@	�PK�@�PK�@!�PK�@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���$!2@��r��y?1��D�@I�[��,@Y|G�	1��?r0*	�� �r�s@2U
Iterator::Model::ParallelMapV26��\�?!�e�ؿ�H@)6��\�?1�e�ؿ�H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath��bE�?!�9���)5@)��v�
�?1�/t�W2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj�L�:�?![�~K��.@)���=^�?11�/ߥ$@:Preprocessing2F
Iterator::ModelQk�w���?!��;P�L@)��I��4�?1.�ڋA @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipo+�6+�?!I�gį	E@)�Ù_͑?1QZ����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���Ss��?!U����@)���Ss��?1U����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoro�$�j�?!sQI��@)o�$�j�?1sQI��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�77.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�PK�@Ii�5[S@Q�_�K.@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��r��y?��r��y?!��r��y?      ��!       "	��D�@��D�@!��D�@*      ��!       2      ��!       :	�[��,@�[��,@!�[��,@B      ��!       J	|G�	1��?|G�	1��?!|G�	1��?R      ��!       Z	|G�	1��?|G�	1��?!|G�	1��?b      ��!       JGPUY�PK�@b qi�5[S@y�_�K.@