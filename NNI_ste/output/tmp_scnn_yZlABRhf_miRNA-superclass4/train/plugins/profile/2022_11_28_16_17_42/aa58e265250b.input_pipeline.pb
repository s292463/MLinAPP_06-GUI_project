	8K�rNL@8K�rNL@!8K�rNL@	 �Y,U��? �Y,U��?! �Y,U��?"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails08K�rNL@Q��9��?1�8�~ߓ0@Ix�=\�C@Y|�&��?r0*	�Q�E�@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceb�����@!a�m 2X@)b�����@1a�m 2X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�9}=_�?!���+Az�?)0c
�8��?1�FN[��?:Preprocessing2U
Iterator::Model::ParallelMapV2�8c���?!�VU���?)�8c���?1�VU���?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�"�J @!T��?4TX@)�3�c�=�?1���	�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�n���N@!����X@)�����?1O��oe��?:Preprocessing2F
Iterator::Model����^��?!��Sy��?)L5���?1�]B;�P�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorJ����?!��%���?)J����?1��%���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�70.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 �Y,U��?I�^ZՅQ@Q	�3M�H=@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Q��9��?Q��9��?!Q��9��?      ��!       "	�8�~ߓ0@�8�~ߓ0@!�8�~ߓ0@*      ��!       2      ��!       :	x�=\�C@x�=\�C@!x�=\�C@B      ��!       J	|�&��?|�&��?!|�&��?R      ��!       Z	|�&��?|�&��?!|�&��?b      ��!       JGPUY �Y,U��?b q�^ZՅQ@y	�3M�H=@