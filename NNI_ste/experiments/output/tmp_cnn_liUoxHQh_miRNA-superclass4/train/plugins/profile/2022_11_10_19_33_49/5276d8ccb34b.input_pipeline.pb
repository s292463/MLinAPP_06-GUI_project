	��d��@��d��@!��d��@	srE@srE@!srE@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��d��@�7L4H��?1�Q,���?A,}����?I:� U\@Y�=ϟ6��?rEagerKernelExecute 0*	J�z��]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\;Qi�?!;sw�f�?@)g`�eM,�?1����\�:@:Preprocessing2F
Iterator::Model;�� �>�?!��h�Z�C@)2Xq��0�?1\EI�h6@:Preprocessing2U
Iterator::Model::ParallelMapV2D�ÖM�?!�����1@)D�ÖM�?1�����1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq:Ɇ?!��b�"@)q:Ɇ?1��b�"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatev()� �?!T��"2@)�;�%8�?1눯"�|!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�o%;6�?!w��N@)��4�?1f��om@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��fc%�y?!T�(X@)��fc%�y?1T�(X@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��8~�?!0�Ԉ�/4@)
�F�c?1V�R�Tk @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�50.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9trE@Iiΰ�	RR@QX���48@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�7L4H��?�7L4H��?!�7L4H��?      ��!       "	�Q,���?�Q,���?!�Q,���?*      ��!       2	,}����?,}����?!,}����?:	:� U\@:� U\@!:� U\@B      ��!       J	�=ϟ6��?�=ϟ6��?!�=ϟ6��?R      ��!       Z	�=ϟ6��?�=ϟ6��?!�=ϟ6��?b      ��!       JGPUYtrE@b qiΰ�	RR@yX���48@