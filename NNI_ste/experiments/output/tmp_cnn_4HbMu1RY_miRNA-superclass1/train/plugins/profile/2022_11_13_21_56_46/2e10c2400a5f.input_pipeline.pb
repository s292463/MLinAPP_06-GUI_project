	�"��]�@�"��]�@!�"��]�@	v�� 0��?v�� 0��?!v�� 0��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�"��]�@.�;1�E�?1�Sͬ��?AV�F�?�?I^H��0~@Yj�~�^��?rEagerKernelExecute 0*	��~j��^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat5s����?!�t�$�@@)�A�F���?1-R�M<@:Preprocessing2F
Iterator::Model�r���?!1
JWD@)������?1~�[d�7@:Preprocessing2U
Iterator::Model::ParallelMapV2�z�L�x�?!��\8J`0@)�z�L�x�?1��\8J`0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�b��^'�?!.�m*�0@)��A�f�?1�G�O�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�B���?!؊p�j� @)�B���?1؊p�j� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'��n��?!������M@)�F��ҁ?1��+�b�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�5Φ#�{?!$W����@)�5Φ#�{?1$W����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�$xC�?!߉���C3@)=���mg?1���#�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 10.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�57.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9v�� 0��?I�S*Q@QeؠU\�=@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	.�;1�E�?.�;1�E�?!.�;1�E�?      ��!       "	�Sͬ��?�Sͬ��?!�Sͬ��?*      ��!       2	V�F�?�?V�F�?�?!V�F�?�?:	^H��0~@^H��0~@!^H��0~@B      ��!       J	j�~�^��?j�~�^��?!j�~�^��?R      ��!       Z	j�~�^��?j�~�^��?!j�~�^��?b      ��!       JGPUYv�� 0��?b q�S*Q@yeؠU\�=@