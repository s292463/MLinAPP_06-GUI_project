	�-y<�7@�-y<�7@!�-y<�7@	Z�!�1��?Z�!�1��?!Z�!�1��?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�-y<�7@�:8؛�?1��fC5@A$D��b?I���?Y��£�?rEagerKernelExecute 0*	z�&1�d@2F
Iterator::ModelxE��?!�D|k�H@)�1^�?1�c�Z��?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{��ɪ?!'\C�?@)kg{��?1���;@:Preprocessing2U
Iterator::Model::ParallelMapV2՗���˝?!C5�8�1@)՗���˝?1C5�8�1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Ҩ�Ɇ?!T����@)�Ҩ�Ɇ?1T����@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��S�*�?!\fg��(@)���͋�?1eP� Y�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#,*�t��?!�3���aI@)1A�º�?1{.�0�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��ң�~?!��R��	@)��ң�~?1��R��	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�mUٗ?!-�N�,@)�0Xre?1�6F�;�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Z�!�1��?I �wԽ� @Qk��~#�V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�:8؛�?�:8؛�?!�:8؛�?      ��!       "	��fC5@��fC5@!��fC5@*      ��!       2	$D��b?$D��b?!$D��b?:	���?���?!���?B      ��!       J	��£�?��£�?!��£�?R      ��!       Z	��£�?��£�?!��£�?b      ��!       JGPUYZ�!�1��?b q �wԽ� @yk��~#�V@