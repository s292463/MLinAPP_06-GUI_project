	B\9{�y@B\9{�y@!B\9{�y@	�����?�����?!�����?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLB\9{�y@��v��B�?1X�<׷�v@A�&����?IC �8�FD@Y�r۾G}�?rEagerKernelExecute 0*	�ʡE��}@2F
Iterator::Model\[%X�?!r���.�S@)��A$C�?1X}��!R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate'l?�ì?!��pA�m'@)����i2�?1o�$f�&&@:Preprocessing2U
Iterator::Model::ParallelMapV2���5>��?!���̊@)���5>��?1���̊@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���RxМ?!V����w@)^.�;1�?1��=M$�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(���%V�?!7f��Ds5@)�'eRC�?1����<
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�W\�{?!UN�*w�?)�W\�{?1UN�*w�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap3�Vzm6�?!T�Pus�(@)� w�(g?1�5>���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�'�>�Y?!t���)��?)�'�>�Y?1t���)��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicel#�	�X?!4���q�?)l#�	�X?14���q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�����?I�S�}l�$@Q��P-_V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��v��B�?��v��B�?!��v��B�?      ��!       "	X�<׷�v@X�<׷�v@!X�<׷�v@*      ��!       2	�&����?�&����?!�&����?:	C �8�FD@C �8�FD@!C �8�FD@B      ��!       J	�r۾G}�?�r۾G}�?!�r۾G}�?R      ��!       Z	�r۾G}�?�r۾G}�?!�r۾G}�?b      ��!       JGPUY�����?b q�S�}l�$@y��P-_V@