	�4f�@�4f�@!�4f�@	�I�'@�I�'@!�I�'@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�4f�@�Qԙ{H�?1�|^�4@A�X��+��?I�8~�t@Yɓ�k&��?rEagerKernelExecute 0*	��Mbs@2U
Iterator::Model::ParallelMapV2��I��?!���~׺J@)��I��?1���~׺J@:Preprocessing2F
Iterator::Model����ދ�?!�W0ҖQ@)�sCSv�?1���Ù�0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat!����=�?!O.�W��2@)Y���j�?1����:0@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�=����?!��uPZ@)�=����?1��uPZ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipPU��X6�?!��>��=@)������?1lj9Q�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��;�2�?!�5��@)K����h�?1.q��;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor<��kЗ~?!��
lֈ@)<��kЗ~?1��
lֈ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap)B�v���?!�˽��@)��	�yk?1� �O�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�34.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�I�'@I�2/���H@Q�xp'��G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Qԙ{H�?�Qԙ{H�?!�Qԙ{H�?      ��!       "	�|^�4@�|^�4@!�|^�4@*      ��!       2	�X��+��?�X��+��?!�X��+��?:	�8~�t@�8~�t@!�8~�t@B      ��!       J	ɓ�k&��?ɓ�k&��?!ɓ�k&��?R      ��!       Z	ɓ�k&��?ɓ�k&��?!ɓ�k&��?b      ��!       JGPUY�I�'@b q�2/���H@y�xp'��G@