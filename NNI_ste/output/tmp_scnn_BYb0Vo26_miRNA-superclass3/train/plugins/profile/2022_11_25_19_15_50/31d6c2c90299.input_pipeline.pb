	�U�Z�.@�U�Z�.@!�U�Z�.@	.�%��@.�%��@!.�%��@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�U�Z�.@��ͪ��v?1�x�0D�@I@�8'@Y�wD���?r0*	��ʡE&d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�}iƲ?!�\��F@)_y��"��?1��l�D@:Preprocessing2F
Iterator::Model�I*S�A�?!V�b�d=@)>w��׹�?18�/�+/@:Preprocessing2U
Iterator::Model::ParallelMapV2�Ҩ�ɖ?!s���q�+@)�Ҩ�ɖ?1s���q�+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��p�5�?!p� X-�4@)�`���?1D��_�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceK?�a�?!��V��#@)K?�a�?1��V��#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�d73��?!)0��<�@)�d73��?1)0��<�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(֩�=#�?!kP�	��Q@)��Dׅ|?1��
�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�75.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9.�%��@I�N���R@Q;�a2\�2@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ͪ��v?��ͪ��v?!��ͪ��v?      ��!       "	�x�0D�@�x�0D�@!�x�0D�@*      ��!       2      ��!       :	@�8'@@�8'@!@�8'@B      ��!       J	�wD���?�wD���?!�wD���?R      ��!       Z	�wD���?�wD���?!�wD���?b      ��!       JGPUY.�%��@b q�N���R@y;�a2\�2@