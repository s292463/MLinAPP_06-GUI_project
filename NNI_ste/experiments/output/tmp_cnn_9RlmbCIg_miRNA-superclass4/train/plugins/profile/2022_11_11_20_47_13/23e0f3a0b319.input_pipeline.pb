	��7�U�t@��7�U�t@!��7�U�t@	��@�ZB�?��@�ZB�?!��@�ZB�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��7�U�t@�2�}��?1K!�K�r@At%�?��?I̘�5��=@YG�I���?rEagerKernelExecute 0*	�E����r@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateW������?!L��L�sQ@))Z���?1�W�	Q@:Preprocessing2F
Iterator::Model��U��6�?!��nѿb4@)	Q����?1}�{%*@:Preprocessing2U
Iterator::Model::ParallelMapV2XU/��d�?!	U��@@)XU/��d�?1	U��@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2 Tq��?!@/��@)DmFA��?1r�B�sI@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-z��y�?!�H�P�S@)�O@��?1	��Xr�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorB&9{z?!���ZHK@)B&9{z?1���ZHK@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�=�N��i?!{כ4��?)�=�N��i?1{כ4��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�|���?!�7����Q@)nYk(�g?1�/J�U��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�J�E�]?!�n�9f�?)�J�E�]?1�n�9f�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��@�ZB�?IX����"@Q���H�V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�2�}��?�2�}��?!�2�}��?      ��!       "	K!�K�r@K!�K�r@!K!�K�r@*      ��!       2	t%�?��?t%�?��?!t%�?��?:	̘�5��=@̘�5��=@!̘�5��=@B      ��!       J	G�I���?G�I���?!G�I���?R      ��!       Z	G�I���?G�I���?!G�I���?b      ��!       JGPUY��@�ZB�?b qX����"@y���H�V@