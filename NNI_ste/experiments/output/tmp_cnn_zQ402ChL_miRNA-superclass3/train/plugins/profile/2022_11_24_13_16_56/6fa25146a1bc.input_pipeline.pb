	����Cz@����Cz@!����Cz@	P5����?P5����?!P5����?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL����Cz@�=yX��?1Ʀ�B rw@A2�CP5�?IB�f���E@Y�����?rEagerKernelExecute 0*	�V�r@2U
Iterator::Model::ParallelMapV2p�DIH��?!JL���K@)p�DIH��?1JL���K@:Preprocessing2F
Iterator::Model9��� �?!��Pm�)R@)#��2R�?1�����0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::ConcatenateEdX���?!��h=�z.@)B�V�9Υ?10~��),@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�P�,�?!�q�J
Y;@)q��]P�?1��Y?�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8�T���?!,�3`z@@)���/�?1��m���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-@�j�y?!&/�i�� @)-@�j�y?1&/�i�� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+Q��r��?!�)��h�/@)W��mUb?1w�����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceW!�'�>]?!�3_h���?)W!�'�>]?1�3_h���?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor�P�l\?!��L6(�?)�P�l\?1��L6(�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9P5����?I�v�dGf%@Q�`4>QV@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�=yX��?�=yX��?!�=yX��?      ��!       "	Ʀ�B rw@Ʀ�B rw@!Ʀ�B rw@*      ��!       2	2�CP5�?2�CP5�?!2�CP5�?:	B�f���E@B�f���E@!B�f���E@B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?b      ��!       JGPUYP5����?b q�v�dGf%@y�`4>QV@